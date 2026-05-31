"""
RF20VL Few-Shot Object Detection — Multi-GPU DDP Inference & Evaluation

Usage:
    torchrun --nproc_per_node=5 inference_ddp.py --split test
    torchrun --nproc_per_node=5 inference_ddp.py --split test --subset gwhd2021-fsod-atsv
    torchrun --nproc_per_node=1 inference_ddp.py --split valid  # 单卡等价于原脚本

Pipeline (per rank):
  1. Initialize distributed process group (NCCL)
  2. Load Sam3Segmenter on local GPU
  3. For each subset, shard samples across ranks (round-robin)
  4. Run inference on local shard
  5. Save local results to temp pkl
  6. Rank 0 merges all results, runs COCO evaluation, saves submissions
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.rf20vl import DatasetRF20VL, get_all_subsets
from model.sam3 import Sam3Segmenter
from model.vqa_rescore import VQARescorer
from sam3.sam3.model.box_ops import box_xywh_to_xyxy, box_xyxy_to_xywh


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(results_dir, rank, debug=False):
    """Configure logging — only rank 0 writes to file + console."""
    os.makedirs(results_dir, exist_ok=True)

    logger = logging.getLogger(f"inference-ddp-r{rank}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    level = logging.DEBUG if debug else logging.INFO

    if rank == 0:
        log_path = os.path.join(results_dir, "inference.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter(f"[GPU{rank}] %(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _collate_fn(batch):
    return batch[0]


def load_class_descriptions(prompt_dir, subset):
    """加载指定 subset 的 class description 映射。"""
    if not os.path.isdir(prompt_dir):
        return {}
    best_match = None
    for fname in os.listdir(prompt_dir):
        if not fname.startswith("all_refined_class_instructions_") or not fname.endswith(".json"):
            continue
        base = fname[len("all_refined_class_instructions_"):-len(".json")]
        if subset.startswith(base) and (best_match is None or len(base) > len(best_match)):
            best_match = base
    if best_match is None:
        return {}
    json_path = os.path.join(prompt_dir, f"all_refined_class_instructions_{best_match}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_mapping(mapping_dir, subset):
    """加载 prompt mapping 文件，返回 {category: best_prompt}。

    对每个类别选取 prompt_score 最高的候选 prompt。
    如果文件不存在则返回空 dict，不影响默认行为。
    """
    mapping_path = os.path.join(mapping_dir, f"{subset}.json")
    if not os.path.isfile(mapping_path):
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for cat_name, candidates in data.items():
        best_prompt = cat_name
        best_score = -1.0
        for prompt, metrics in candidates.items():
            score = metrics.get("prompt_score", 0.0)
            if score > best_score:
                best_score = score
                best_prompt = prompt
        result[cat_name] = best_prompt
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def concat_images(supp_pil, query_pil):
    """Horizontally concatenate support (left) + query (right)."""
    qw, qh = query_pil.size
    sw, sh = supp_pil.size

    if sh != qh:
        scale = qh / sh
        new_sw = int(sw * scale)
        supp_pil = supp_pil.resize((new_sw, qh), Image.LANCZOS)
    else:
        scale = 1.0
        new_sw = sw

    cat_w = new_sw + qw
    cat_pil = Image.new("RGB", (cat_w, qh))
    cat_pil.paste(supp_pil, (0, 0))
    cat_pil.paste(query_pil, (new_sw, 0))
    return cat_pil, new_sw, scale


def scale_bboxes(bboxes_xywh, scale):
    """Scale xywh bboxes by a factor."""
    if bboxes_xywh.numel() == 0:
        return bboxes_xywh.clone()
    scaled = bboxes_xywh.clone()
    scaled[:, 0] *= scale
    scaled[:, 1] *= scale
    scaled[:, 2] *= scale
    scaled[:, 3] *= scale
    return scaled


def _draw_boxes(image_pil, boxes_xyxy, color, width=3):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img


def save_visualization(results_dir, category, query_img_path, query_pil,
                       gt_xyxy, result, support_items):
    query_stem = os.path.splitext(os.path.basename(query_img_path))[0]

    for pt in ["visual", "tv", "text"]:
        boxes = result[f"{pt}_boxes"].cpu()
        vis_dir = os.path.join(results_dir, "vis", pt, category)
        os.makedirs(vis_dir, exist_ok=True)

        query_vis = _draw_boxes(query_pil, gt_xyxy, color="green", width=3)
        if boxes.numel() > 0:
            query_vis = _draw_boxes(query_vis, boxes, color="red", width=3)
        query_vis.save(os.path.join(vis_dir, f"{query_stem}.jpg"), quality=90)

    # support images (shared, only save once)
    supp_dir = os.path.join(results_dir, "vis", category, "supports")
    os.makedirs(supp_dir, exist_ok=True)
    for supp_path, supp_bboxes_xywh in support_items:
        supp_pil = Image.open(supp_path).convert("RGB")
        supp_xyxy = box_xywh_to_xyxy(supp_bboxes_xywh)
        supp_vis = _draw_boxes(supp_pil, supp_xyxy, color="green", width=3)
        supp_save_name = os.path.splitext(os.path.basename(supp_path))[0] + ".jpg"
        supp_vis.save(os.path.join(supp_dir, supp_save_name), quality=90)


# ---------------------------------------------------------------------------
# Submission & Evaluation
# ---------------------------------------------------------------------------

class SubmissionCollector:
    def __init__(self, pred_types):
        self.pred_types = pred_types
        self.submissions = {pt: defaultdict(lambda: defaultdict(list)) for pt in pred_types}
        self.all_img_ids = defaultdict(set)

    def add(self, pred_type, subset, img_id, cat_id, boxes_xyxy, scores):
        self.all_img_ids[subset].add(img_id)
        if boxes_xyxy.numel() == 0:
            # 添加一个极低分数的 dummy bbox，避免评测因空 instance 列表报错
            self.submissions[pred_type][subset][img_id].append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "score": 1e-6,
            })
            return
        boxes_xywh = box_xyxy_to_xywh(boxes_xyxy)
        for i in range(boxes_xywh.shape[0]):
            self.submissions[pred_type][subset][img_id].append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": boxes_xywh[i].numpy().astype(np.float64),
                "score": float(scores[i].item()),
            })

    def save_all(self, save_dir):
        for pt in self.pred_types:
            pt_dir = os.path.join(save_dir, pt)
            os.makedirs(pt_dir, exist_ok=True)
            for subset in sorted(self.all_img_ids.keys()):
                img_dict = self.submissions[pt][subset]
                submission_list = []
                for img_id in sorted(self.all_img_ids[subset]):
                    submission_list.append({
                        "image_id": int(img_id),
                        "instances": img_dict[img_id],
                    })
                pkl_path = os.path.join(pt_dir, f"{subset}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(submission_list, f)

    def get_flat_predictions(self, pred_type, subset):
        flat = []
        for img_id in sorted(self.submissions[pred_type][subset].keys()):
            flat.extend(self.submissions[pred_type][subset][img_id])
        return flat


def coco_evaluate(ann_json_path, coco_predictions):
    if not coco_predictions:
        return None

    for pred in coco_predictions:
        if isinstance(pred["bbox"], np.ndarray):
            pred["bbox"] = pred["bbox"].tolist()

    coco_gt = COCO(ann_json_path)
    coco_dt = coco_gt.loadRes(coco_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": float(coco_eval.stats[0]),
        "AP_50": float(coco_eval.stats[1]),
        "AP_75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RF20VL FSOD DDP Inference & Evaluation")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Path to RF20VL data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "valid"], help="Query split")
    parser.add_argument("--subset", type=str, default=None,
                        help="Evaluate only this subset (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="NMS IoU threshold for candidate box dedup")
    parser.add_argument("--pred-nms-iou", type=float, default=0.5,
                        help="NMS IoU threshold for final prediction dedup")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples per subset (for quick testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override results output directory")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save predictions as JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualization saving")
    parser.add_argument("--vqa-rescore", action="store_true",
                        help="Apply VQA rescoring to predictions")
    parser.add_argument("--vqa-crop", action="store_true",
                        help="VQA uses cropped region (bbox expanded by 50%%) instead of full image")
    parser.add_argument("--vqa-target", type=str, default="tv",
                        choices=["tv", "text", "visual"],
                        help="Which prediction type to apply VQA rescoring to (default: tv)")
    parser.add_argument("--vqa-prompt-dir", type=str,
                        default="DetPO/prompts/detpo/Qwen3-VL-8B-Instruct",
                        help="Directory containing class description JSON files")
    parser.add_argument("--prompt-mapping-dir", type=str,
                        default="prompts/sam3_prompt_mapping",
                        help="Directory containing SAM3 prompt mapping JSON files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Distributed setup ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = (rank == 0)

    # --- Results directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        results_dir = args.output_dir
    else:
        subset_tag = args.subset if args.subset else "all"
        results_dir = os.path.join("results", f"{timestamp}_{subset_tag}_{args.split}")

    log = setup_logging(results_dir, rank, debug=args.debug)

    if is_main:
        log.info("=" * 60)
        log.info("RF20VL FSOD — DDP Inference (%d GPUs)", world_size)
        log.info("=" * 60)
        log.info("Arguments: %s", vars(args))

    # --- Model loading ---
    t0_model = time.time()
    log.info("Loading Sam3Segmenter on %s ...", device)
    segmenter = Sam3Segmenter(device=device)
    t_model = time.time() - t0_model
    log.info("Model loaded in %.2f s", t_model)

    # --- VQA Rescorer ---
    vqa_rescorer = None
    if args.vqa_rescore:
        vqa_rescorer = VQARescorer(
            api_key="EMPTY",
            server_url="http://localhost:22002/v1",
            model_name="qwen3-vl-8b",
            use_crop=args.vqa_crop,
        )
        log.info("VQA Rescorer initialized (crop=%s)", args.vqa_crop)

    # --- Dataset ---
    subsets = get_all_subsets(args.data_path)
    if args.subset:
        subsets = [s for s in subsets if s == args.subset]
    if is_main:
        log.info("Found %d subset(s): %s", len(subsets), subsets)

    pred_types = ["visual", "tv", "text"]
    if args.vqa_rescore:
        pred_types.append("vqa")

    # Per-rank local collection
    local_preds = []       # [(pt, subset, img_id, cat_id, boxes_list, scores_list), ...]
    local_sample_times = []
    local_all_img_ids = defaultdict(set)

    t0_total = time.time()

    for subset in subsets:
        t0_ds = time.time()
        dataset = DatasetRF20VL(args.data_path, subset, query_split=args.split,
                                max_samples=args.max_samples)
        t_ds = time.time() - t0_ds

        # Shard indices: rank 0 gets 0,5,10,...; rank 1 gets 1,6,11,...; etc.
        total_samples = len(dataset)
        my_indices = list(range(rank, total_samples, world_size))
        my_count = len(my_indices)

        log.info("[%s] %d total samples, rank %d handles %d (indices %s)",
                 subset, total_samples, rank, my_count,
                 f"{my_indices[0]}..{my_indices[-1]}" if my_indices else "none")

        class_descriptions = {}
        if vqa_rescorer is not None:
            class_descriptions = load_class_descriptions(args.vqa_prompt_dir, subset)

        prompt_mapping = load_prompt_mapping(args.prompt_mapping_dir, subset)
        if prompt_mapping:
            log.info("[%s] Loaded prompt mapping: %d categories", subset, len(prompt_mapping))

        for local_idx, global_idx in enumerate(my_indices):
            t0_sample = time.time()
            sample = dataset[global_idx]
            category = sample["category"]
            class_id = sample["class_id"]
            query_w, query_h = sample["query_img_size"]
            _, img_id = dataset.index[global_idx]

            query_pil = Image.open(sample["query_img_path"]).convert("RGB")

            # --- Collect candidate boxes ---
            all_candidate_boxes = []
            support_items = list(zip(
                sample["support_img_paths"],
                sample["support_bboxes"],
            ))

            for si, (supp_path, supp_bboxes_xywh) in enumerate(support_items):
                supp_pil = Image.open(supp_path).convert("RGB")
                cat_pil, offset_x, scale = concat_images(supp_pil, query_pil)
                supp_scaled = scale_bboxes(supp_bboxes_xywh, scale)
                supp_xyxy = box_xywh_to_xyxy(supp_scaled)

                query_boxes = segmenter.get_query_boxes_from_cat(cat_pil, supp_xyxy)

                if query_boxes.numel() > 0:
                    query_boxes[:, 0] = query_boxes[:, 0].clamp(min=0, max=query_w)
                    query_boxes[:, 1] = query_boxes[:, 1].clamp(min=0, max=query_h)
                    query_boxes[:, 2] = query_boxes[:, 2].clamp(min=0, max=query_w)
                    query_boxes[:, 3] = query_boxes[:, 3].clamp(min=0, max=query_h)
                    all_candidate_boxes.append(query_boxes)

            # --- Merge & NMS ---
            if not all_candidate_boxes:
                candidate_boxes = torch.zeros(0, 4, device=device)
            else:
                candidate_boxes = torch.cat(all_candidate_boxes, dim=0)
                if candidate_boxes.numel() > 0:
                    dummy_scores = torch.ones(candidate_boxes.shape[0], device=device)
                    keep = nms(candidate_boxes, dummy_scores, args.nms_iou)
                    candidate_boxes = candidate_boxes[keep]

            # --- Inference ---
            prompt = prompt_mapping.get(category, category)
            result = segmenter.forward_with_query_boxes(
                query_pil, candidate_boxes, prompt=prompt,
            )

            vqa_target = args.vqa_target
            target_boxes = None
            target_scores = None
            for pt in ["visual", "tv", "text"]:
                boxes = result[f"{pt}_boxes"].cpu()
                scores = result[f"{pt}_scores"].cpu()
                if boxes.numel() > 0 and args.pred_nms_iou < 1.0:
                    keep = nms(boxes, scores, args.pred_nms_iou)
                    boxes = boxes[keep]
                    scores = scores[keep]

                if pt == vqa_target:
                    target_boxes = boxes
                    target_scores = scores

                local_all_img_ids[subset].add(img_id)
                local_preds.append((
                    pt, subset, img_id, class_id,
                    boxes.numpy().tolist(),
                    scores.numpy().tolist(),
                ))

            # VQA rescoring
            if vqa_rescorer is not None and target_boxes is not None and target_boxes.numel() > 0:
                desc = class_descriptions.get(category)
                boxes_xywh = box_xyxy_to_xywh(target_boxes)
                rescored_scores = []
                for i in range(boxes_xywh.shape[0]):
                    bbox_xywh = boxes_xywh[i].tolist()
                    vqa_score = vqa_rescorer(query_pil, bbox_xywh, category, desc)
                    rescored_scores.append(vqa_score)

                vqa_scores = []
                for i, vs in enumerate(rescored_scores):
                    if vs < 0:
                        vqa_scores.append(float(target_scores[i].item()))
                    else:
                        vqa_scores.append(vs)

                local_preds.append((
                    "vqa", subset, img_id, class_id,
                    target_boxes.numpy().tolist(),
                    vqa_scores,
                ))

            t_sample = time.time() - t0_sample
            local_sample_times.append(t_sample)

            n_vis = result["visual_boxes"].shape[0]
            n_tv = result["tv_boxes"].shape[0]
            n_txt = result["text_boxes"].shape[0]

            log.info("[%s] rank%d %d/%d | %s | supports=%d | "
                     "cands=%d | v=%d tv=%d t=%d | %.2f s",
                     subset, rank, local_idx + 1, my_count, category,
                     len(support_items), candidate_boxes.shape[0],
                     n_vis, n_tv, n_txt, t_sample)

            # --- Visualization ---
            if not args.no_vis:
                gt_xyxy = box_xywh_to_xyxy(sample["query_bboxes"])
                save_visualization(
                    results_dir, category,
                    sample["query_img_path"], query_pil,
                    gt_xyxy,
                    result,
                    support_items,
                )

    # ---- Save local results to temp file ----
    tmp_dir = os.path.join(results_dir, ".tmp_ddp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"rank{rank}.pkl")

    with open(tmp_path, "wb") as f:
        pickle.dump({
            "preds": local_preds,
            "sample_times": local_sample_times,
            "all_img_ids": {k: list(v) for k, v in local_all_img_ids.items()},
        }, f)

    log.info("Local results saved (%d preds, %d samples)", len(local_preds), len(local_sample_times))

    # ---- Synchronize ----
    dist.barrier()

    # ---- Rank 0: merge, evaluate, save ----
    if is_main:
        t_merge_start = time.time()

        submission = SubmissionCollector(pred_types)
        all_sample_times = []

        for r in range(world_size):
            path = os.path.join(tmp_dir, f"rank{r}.pkl")
            with open(path, "rb") as f:
                data = pickle.load(f)

            for pt, subset, img_id, cat_id, boxes_list, scores_list in data["preds"]:
                boxes_t = torch.tensor(boxes_list, dtype=torch.float32)
                scores_t = torch.tensor(scores_list, dtype=torch.float32)
                submission.add(pt, subset, img_id, cat_id, boxes_t, scores_t)

            all_sample_times.extend(data["sample_times"])
            for subset, ids in data["all_img_ids"].items():
                submission.all_img_ids[subset].update(ids)

        log.info("Merged results from %d ranks in %.2f s", world_size, time.time() - t_merge_start)

        # --- COCO evaluation per subset ---
        coco_results = {}
        for subset in subsets:
            ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")
            if os.path.isfile(ann_json):
                log.info("--- COCO Evaluation: %s ---", subset)
                coco_results[subset] = {}
                for pt in pred_types:
                    flat_preds = submission.get_flat_predictions(pt, subset)
                    metrics = coco_evaluate(ann_json, flat_preds)
                    if metrics:
                        coco_results[subset][pt] = metrics
                        log.info("  [%s] AP=%.4f  AP50=%.4f  AP75=%.4f  "
                                 "AP_s=%.4f  AP_m=%.4f  AP_l=%.4f  "
                                 "AR_1=%.4f  AR_10=%.4f  AR_100=%.4f",
                                 pt, metrics["AP"], metrics["AP_50"], metrics["AP_75"],
                                 metrics["AP_small"], metrics["AP_medium"], metrics["AP_large"],
                                 metrics["AR_1"], metrics["AR_10"], metrics["AR_100"])
                    else:
                        log.info("  [%s] No predictions for COCO eval.", pt)
            else:
                log.warning("Annotation file not found: %s", ann_json)

        t_total = time.time() - t0_total

        # --- Timing summary ---
        log.info("=" * 60)
        log.info("TIMING SUMMARY")
        log.info("=" * 60)
        log.info("Model loading:    %.2f s", t_model)
        log.info("Total inference:  %.2f s", t_total)
        log.info("GPUs used:        %d", world_size)
        log.info("Samples processed: %d", len(all_sample_times))
        if all_sample_times:
            log.info("Avg per sample:   %.3f s", np.mean(all_sample_times))
            log.info("Min per sample:   %.3f s", np.min(all_sample_times))
            log.info("Max per sample:   %.3f s", np.max(all_sample_times))

        # --- Save submission pickle files ---
        submission_dir = os.path.join(results_dir, "submissions")
        submission.save_all(submission_dir)
        log.info("Submission pickle files saved to %s", submission_dir)

        # --- Save JSON ---
        output_json = args.output_json or os.path.join(results_dir, "results.json")
        out = {
            "timestamp": timestamp,
            "subsets": subsets,
            "split": args.split,
            "world_size": world_size,
            "args": vars(args),
            "timing": {
                "model_load_s": round(t_model, 2),
                "total_inference_s": round(t_total, 2),
                "num_samples": len(all_sample_times),
                "avg_per_sample_s": round(float(np.mean(all_sample_times)), 3) if all_sample_times else 0,
                "min_per_sample_s": round(float(np.min(all_sample_times)), 3) if all_sample_times else 0,
                "max_per_sample_s": round(float(np.max(all_sample_times)), 3) if all_sample_times else 0,
            },
            "coco_results": {},
        }
        for subset_name, pt_metrics in coco_results.items():
            out["coco_results"][subset_name] = {}
            for pt, metrics in pt_metrics.items():
                out["coco_results"][subset_name][pt] = {
                    k: round(v, 6) for k, v in metrics.items()
                }
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        log.info("Results saved to %s", output_json)

        # --- Cleanup temp files ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("Log file: %s", os.path.join(results_dir, "inference.log"))

    # ---- Final sync and cleanup ----
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
