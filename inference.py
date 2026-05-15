"""
RF20VL Few-Shot Object Detection — Inference & Evaluation Script

Pipeline:
  1. Create Sam3Segmenter
  2. For each subset & sample:
     - Concatenate each support image with query image
     - Collect query-side candidate boxes from all supports
     - Run forward_with_query_boxes (text=category, bboxes=candidates)
  3. Compute AP@0.5 / AP@0.75 per class, then mAP
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from torch.utils.data import DataLoader
from datetime import datetime

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.ops import nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.rf20vl import DatasetRF20VL, get_all_subsets
from model.sam3 import Sam3Segmenter
from sam3.sam3.model.box_ops import box_xywh_to_xyxy, box_xyxy_to_xywh


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(results_dir, debug=False):
    """Configure logging to both console and file."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "inference.log")

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    level = logging.DEBUG if debug else logging.INFO

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _collate_fn(batch):
    """Return items as-is (no stacking) — dataset has variable-length fields."""
    return batch[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def concat_images(supp_pil, query_pil):
    """Horizontally concatenate support (left) + query (right).

    If heights differ, resize support to match query height (keep aspect ratio).

    Returns:
        cat_pil: concatenated PIL image
        offset_x: width of the (resized) support image (query half starts here)
        scale: resize scale factor applied to support image
    """
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
    """Scale xywh bboxes by a factor (for resized images)."""
    if bboxes_xywh.numel() == 0:
        return bboxes_xywh.clone()
    scaled = bboxes_xywh.clone()
    scaled[:, 0] *= scale
    scaled[:, 1] *= scale
    scaled[:, 2] *= scale
    scaled[:, 3] *= scale
    return scaled


def _draw_boxes(image_pil, boxes_xyxy, color, width=3):
    """在 PIL 图像上画框，返回新图像。"""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img


def save_visualization(results_dir, category, query_img_path, query_pil,
                       gt_xyxy, tv_boxes, support_items):
    """保存 query 和 support 的可视化结果。"""
    vis_dir = os.path.join(results_dir, "vis", category)
    os.makedirs(vis_dir, exist_ok=True)

    # --- Query: GT 绿 + pred 红 ---
    query_vis = _draw_boxes(query_pil, gt_xyxy, color="green", width=3)
    if tv_boxes.numel() > 0:
        query_vis = _draw_boxes(query_vis, tv_boxes, color="red", width=3)

    query_stem = os.path.splitext(os.path.basename(query_img_path))[0]
    query_save_name = f"{query_stem}.jpg"
    query_vis.save(os.path.join(vis_dir, query_save_name), quality=90)

    # --- Support ---
    supp_dir = os.path.join(vis_dir, "supports")
    os.makedirs(supp_dir, exist_ok=True)
    for supp_path, supp_bboxes_xywh in support_items:
        supp_pil = Image.open(supp_path).convert("RGB")
        supp_xyxy = box_xywh_to_xyxy(supp_bboxes_xywh)
        supp_vis = _draw_boxes(supp_pil, supp_xyxy, color="green", width=3)
        supp_save_name = os.path.splitext(os.path.basename(supp_path))[0] + ".jpg"
        supp_vis.save(os.path.join(supp_dir, supp_save_name), quality=90)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

class SubmissionCollector:
    """Collect predictions per image for submission pickle files."""

    def __init__(self, pred_types):
        self.pred_types = pred_types
        # submissions[pred_type][subset][img_id] = [instance_dict, ...]
        self.submissions = {pt: defaultdict(lambda: defaultdict(list)) for pt in pred_types}
        # Track all img_ids seen per subset so empty-prediction images are not lost
        self.all_img_ids = defaultdict(set)

    def add(self, pred_type, subset, img_id, cat_id, boxes_xyxy, scores):
        """Convert xyxy predictions to COCO xywh and store."""
        self.all_img_ids[subset].add(img_id)
        if boxes_xyxy.numel() == 0:
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
        """Save pickle files for all pred_types and subsets."""
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
        """Return flat list of COCO-format instance dicts for pycocotools."""
        flat = []
        for img_id in sorted(self.submissions[pred_type][subset].keys()):
            flat.extend(self.submissions[pred_type][subset][img_id])
        return flat


def coco_evaluate(ann_json_path, coco_predictions):
    """Run COCOeval on predictions against GT annotations.

    Args:
        ann_json_path: path to _annotations.coco.json
        coco_predictions: flat list of {image_id, category_id, bbox, score}

    Returns:
        dict with COCO metrics, or None if evaluation fails.
    """
    if not coco_predictions:
        return None

    # Convert numpy bbox arrays to lists for pycocotools
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
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RF20VL FSOD Inference & Evaluation")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Path to RF20VL data directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "valid"], help="Query split")
    parser.add_argument("--subset", type=str, default=None,
                        help="Evaluate only this subset (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nms-iou", type=float, default=0.8,
                        help="NMS IoU threshold for candidate box dedup")
    parser.add_argument("--pred-nms-iou", type=float, default=0.8,
                        help="NMS IoU threshold for final prediction dedup")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples per subset (for quick testing)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save predictions as JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG level logging (file & console)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualization saving")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Results directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subset_tag = args.subset if args.subset else "all"
    results_dir = os.path.join("results", f"{timestamp}_{subset_tag}_{args.split}")
    log = setup_logging(results_dir, debug=args.debug)

    log.info("=" * 60)
    log.info("RF20VL Few-Shot Object Detection — Inference")
    log.info("=" * 60)
    log.info("Arguments: %s", vars(args))

    # --- Model loading ---
    t0_model = time.time()
    log.info("Loading Sam3Segmenter on %s ...", args.device)
    segmenter = Sam3Segmenter(device=args.device)
    t_model = time.time() - t0_model
    log.info("Model loaded in %.2f s", t_model)

    # --- Dataset ---
    subsets = get_all_subsets(args.data_path)
    if args.subset:
        subsets = [s for s in subsets if s == args.subset]
    log.info("Found %d subset(s): %s", len(subsets), subsets)

    pred_types = ["visual", "tv", "text"]
    submission = SubmissionCollector(pred_types)
    coco_results = {}  # coco_results[subset][pt] = metrics dict

    # Per-sample timing
    sample_times = []

    t0_total = time.time()

    for subset in subsets:
        t0_ds = time.time()
        dataset = DatasetRF20VL(args.data_path, subset, query_split=args.split,
                                max_samples=args.max_samples)
        t_ds = time.time() - t0_ds
        log.info("Dataset loaded: subset=%s, samples=%d, categories=%s (%.2f s)",
                 subset, len(dataset), dataset.categories, t_ds)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=_collate_fn)

        for idx, sample in enumerate(dataloader):
            t0_sample = time.time()
            category = sample["category"]
            class_id = sample["class_id"]
            query_w, query_h = sample["query_img_size"]
            _, img_id = dataset.index[idx]

            # --- Load query image as PIL ---
            query_pil = Image.open(sample["query_img_path"]).convert("RGB")

            # --- Collect candidate boxes from each support ---
            all_candidate_boxes = []
            support_items = list(zip(
                sample["support_img_paths"],
                sample["support_bboxes"],
            ))

            for si, (supp_path, supp_bboxes_xywh) in enumerate(support_items):
                supp_pil = Image.open(supp_path).convert("RGB")

                cat_pil, offset_x, scale = concat_images(supp_pil, query_pil)
                log.debug("  support[%d]: %s -> cat_size=%s, scale=%.3f",
                          si, os.path.basename(supp_path), cat_pil.size, scale)

                supp_scaled = scale_bboxes(supp_bboxes_xywh, scale)
                supp_xyxy = box_xywh_to_xyxy(supp_scaled)

                query_boxes = segmenter.get_query_boxes_from_cat(cat_pil, supp_xyxy)
                log.debug("  support[%d]: query_boxes=%d", si, query_boxes.shape[0] if query_boxes.numel() > 0 else 0)

                if query_boxes.numel() > 0:
                    query_boxes[:, 0] = query_boxes[:, 0].clamp(min=0, max=query_w)
                    query_boxes[:, 1] = query_boxes[:, 1].clamp(min=0, max=query_h)
                    query_boxes[:, 2] = query_boxes[:, 2].clamp(min=0, max=query_w)
                    query_boxes[:, 3] = query_boxes[:, 3].clamp(min=0, max=query_h)
                    all_candidate_boxes.append(query_boxes)

            # --- Merge & NMS ---
            if not all_candidate_boxes:
                candidate_boxes = torch.zeros(0, 4, device=segmenter.device)
            else:
                candidate_boxes = torch.cat(all_candidate_boxes, dim=0)
                if candidate_boxes.numel() > 0:
                    dummy_scores = torch.ones(candidate_boxes.shape[0], device=candidate_boxes.device)
                    keep = nms(candidate_boxes, dummy_scores, args.nms_iou)
                    candidate_boxes = candidate_boxes[keep]

            # --- Inference ---
            result = segmenter.forward_with_query_boxes(
                query_pil, candidate_boxes, prompt=category,
            )

            for pt in pred_types:
                boxes = result[f"{pt}_boxes"].cpu()
                scores = result[f"{pt}_scores"].cpu()
                if boxes.numel() > 0 and args.pred_nms_iou < 1.0:
                    keep = nms(boxes, scores, args.pred_nms_iou)
                    boxes = boxes[keep]
                    scores = scores[keep]
                submission.add(pt, subset, img_id, class_id, boxes, scores)

            t_sample = time.time() - t0_sample
            sample_times.append(t_sample)

            n_vis = result["visual_boxes"].shape[0]
            n_tv = result["tv_boxes"].shape[0]
            n_txt = result["text_boxes"].shape[0]

            log.info("[%s] sample %d/%d | category=%s | supports=%d | "
                     "candidates=%d | visual=%d tv=%d text=%d | %.2f s",
                     subset, idx + 1, len(dataloader), category,
                     len(support_items), candidate_boxes.shape[0],
                     n_vis, n_tv, n_txt, t_sample)

            # --- Visualization ---
            if not args.no_vis:
                gt_xyxy = box_xywh_to_xyxy(sample["query_bboxes"])
                save_visualization(
                    results_dir, category,
                    sample["query_img_path"], query_pil,
                    gt_xyxy,
                    result["tv_boxes"].cpu(),
                    support_items,
                )

        # --- COCO evaluation per subset ---
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
    log.info("")
    log.info("--- Timing ---")
    log.info("Model loading:    %.2f s", t_model)
    log.info("Total inference:  %.2f s", t_total)
    log.info("Samples processed: %d", len(sample_times))
    if sample_times:
        log.info("Avg per sample:   %.3f s", np.mean(sample_times))
        log.info("Min per sample:   %.3f s", np.min(sample_times))
        log.info("Max per sample:   %.3f s", np.max(sample_times))

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
        "args": vars(args),
        "timing": {
            "model_load_s": round(t_model, 2),
            "total_inference_s": round(t_total, 2),
            "num_samples": len(sample_times),
            "avg_per_sample_s": round(float(np.mean(sample_times)), 3) if sample_times else 0,
            "min_per_sample_s": round(float(np.min(sample_times)), 3) if sample_times else 0,
            "max_per_sample_s": round(float(np.max(sample_times)), 3) if sample_times else 0,
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

    log.info("Log file: %s", os.path.join(results_dir, "inference.log"))


if __name__ == "__main__":
    main()
