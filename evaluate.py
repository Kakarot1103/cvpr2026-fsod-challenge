"""
RF20VL Few-Shot Object Detection — Standalone Evaluation Script

Evaluate a submission directory against ground-truth COCO annotations.
Usage:
    python evaluate.py --submission submission/20260514_014727_tv
    python evaluate.py --submission submission/20260514_014727_tv --split test
    python evaluate.py --submission submission/20260514_014727_tv --subset gwhd2021-fsod-atsv
"""

import argparse
import json
import os
import pickle
import sys

from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_submission(pkl_path):
    """Load submission pkl and return flat list of COCO-format prediction dicts."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    flat = []
    for item in data:
        for inst in item["instances"]:
            pred = {
                "image_id": inst["image_id"],
                "category_id": inst["category_id"],
                "score": float(inst["score"]),
                "bbox": inst["bbox"].tolist() if isinstance(inst["bbox"], np.ndarray) else list(inst["bbox"]),
            }
            flat.append(pred)
    return flat


def apply_ranking_rescore(predictions, max_score=1.0, min_score=0.1):
    """Reassign scores based on ranking within each (image_id, category_id) group.

    For each group, sort by original score descending, then linearly assign
    scores from max_score (rank 1) to min_score (last rank).
    """
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[(pred["image_id"], pred["category_id"])].append(pred)

    rescored = []
    for key, preds in grouped.items():
        sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)
        n = len(sorted_preds)
        for i, pred in enumerate(sorted_preds):
            if n == 1:
                new_score = max_score
            else:
                new_score = max_score - i * (max_score - min_score) / (n - 1)
            rescored.append({
                "image_id": pred["image_id"],
                "category_id": pred["category_id"],
                "score": float(new_score),
                "bbox": pred["bbox"],
            })
    return rescored


def evaluate_subset(ann_json, predictions):
    """Run COCO evaluation for a single subset.

    Returns dict with COCO metrics, or None on failure.
    """
    if not predictions:
        return None

    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(predictions)
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


def keep_topk_percent(predictions, percent):
    """Keep only top P% predictions per (image_id, category_id) group, sorted by score descending."""
    if percent >= 100:
        return predictions
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[(pred["image_id"], pred["category_id"])].append(pred)
    kept = []
    for key, preds in grouped.items():
        sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)
        n_keep = max(1, int(len(sorted_preds) * percent / 100))
        kept.extend(sorted_preds[:n_keep])
    return kept


def run_ablation_topk(args, pkl_files, submission_dir):
    """Run top-K% ablation study: evaluate with 10%~100% predictions kept."""
    # Load all predictions first
    subset_data = {}
    for pkl_file in pkl_files:
        subset = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(submission_dir, pkl_file)
        ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")
        if not os.path.isfile(ann_json):
            continue
        predictions = load_submission(pkl_path)
        if predictions:
            subset_data[subset] = {"predictions": predictions, "ann_json": ann_json}

    percents = list(range(10, 110, 10))
    ablation_results = {}

    for pct in percents:
        all_metrics = {}
        for subset, data in subset_data.items():
            filtered = keep_topk_percent(data["predictions"], pct)
            m = evaluate_subset(data["ann_json"], filtered)
            if m:
                all_metrics[subset] = m
        if all_metrics:
            mean_ap = np.mean([m["AP"] for m in all_metrics.values()])
            mean_ap50 = np.mean([m["AP_50"] for m in all_metrics.values()])
            mean_ap75 = np.mean([m["AP_75"] for m in all_metrics.values()])
            mean_ar100 = np.mean([m["AR_100"] for m in all_metrics.values()])
            ablation_results[pct] = {
                "mean_AP": float(mean_ap),
                "mean_AP50": float(mean_ap50),
                "mean_AP75": float(mean_ap75),
                "mean_AR_100": float(mean_ar100),
                "num_subsets": len(all_metrics),
            }

    # Print table
    print(f"\n{'TopK%':>6}  {'AP':>8}  {'AP50':>8}  {'AP75':>8}  {'AR100':>8}  {'Subsets':>7}")
    print("-" * 55)
    for pct in percents:
        r = ablation_results.get(pct)
        if r:
            print(f"{pct:>5}%  {r['mean_AP']:>8.4f}  {r['mean_AP50']:>8.4f}  "
                  f"{r['mean_AP75']:>8.4f}  {r['mean_AR_100']:>8.4f}  {r['num_subsets']:>7}")

    # Find best
    best_pct = max(ablation_results, key=lambda p: ablation_results[p]["mean_AP"])
    print(f"\n  Best top-K% = {best_pct}%  (AP={ablation_results[best_pct]['mean_AP']:.4f})")

    return ablation_results


def apply_nms(predictions, iou_threshold):
    """Apply per-(image_id, category_id) NMS with given IoU threshold."""
    import torch
    from torchvision.ops import nms as torchvision_nms

    grouped = defaultdict(list)
    for pred in predictions:
        grouped[(pred["image_id"], pred["category_id"])].append(pred)

    kept = []
    for key, preds in grouped.items():
        if len(preds) <= 1:
            kept.extend(preds)
            continue
        boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float32)
        # xywh -> xyxy
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
        boxes_xyxy[:, 3] += boxes_xyxy[:, 1]
        scores = torch.tensor([p["score"] for p in preds], dtype=torch.float32)
        keep_indices = torchvision_nms(boxes_xyxy, scores, iou_threshold)
        kept.extend([preds[i] for i in keep_indices.tolist()])
    return kept


def run_ablation_nms(args, pkl_files, submission_dir):
    """Run NMS IoU threshold ablation study."""
    import torch

    subset_data = {}
    for pkl_file in pkl_files:
        subset = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(submission_dir, pkl_file)
        ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")
        if not os.path.isfile(ann_json):
            continue
        predictions = load_submission(pkl_path)
        if predictions:
            subset_data[subset] = {"predictions": predictions, "ann_json": ann_json}

    thresholds = [round(t * 0.05, 2) for t in range(10, 19)]  # 0.50, 0.55, ..., 0.90
    ablation_results = {}
    total_preds = sum(len(d["predictions"]) for d in subset_data.values())

    # Baseline: no NMS
    all_metrics = {}
    for subset, data in subset_data.items():
        m = evaluate_subset(data["ann_json"], data["predictions"])
        if m:
            all_metrics[subset] = m
    if all_metrics:
        ablation_results["no_nms"] = {
            "mean_AP": float(np.mean([m["AP"] for m in all_metrics.values()])),
            "mean_AP50": float(np.mean([m["AP_50"] for m in all_metrics.values()])),
            "mean_AP75": float(np.mean([m["AP_75"] for m in all_metrics.values()])),
            "mean_AR_100": float(np.mean([m["AR_100"] for m in all_metrics.values()])),
            "total_kept": total_preds,
            "num_subsets": len(all_metrics),
        }

    for thr in thresholds:
        all_metrics = {}
        total_kept = 0
        for subset, data in subset_data.items():
            filtered = apply_nms(data["predictions"], thr)
            total_kept += len(filtered)
            m = evaluate_subset(data["ann_json"], filtered)
            if m:
                all_metrics[subset] = m
        if all_metrics:
            mean_ap = np.mean([m["AP"] for m in all_metrics.values()])
            mean_ap50 = np.mean([m["AP_50"] for m in all_metrics.values()])
            mean_ap75 = np.mean([m["AP_75"] for m in all_metrics.values()])
            mean_ar100 = np.mean([m["AR_100"] for m in all_metrics.values()])
            ablation_results[thr] = {
                "mean_AP": float(mean_ap),
                "mean_AP50": float(mean_ap50),
                "mean_AP75": float(mean_ap75),
                "mean_AR_100": float(mean_ar100),
                "num_subsets": len(all_metrics),
                "total_kept": total_kept,
                "keep_ratio": round(total_kept / total_preds * 100, 1) if total_preds > 0 else 0,
            }

    # Print table
    print(f"\n{'NMS Thr':>8}  {'AP':>8}  {'AP50':>8}  {'AP75':>8}  {'AR100':>8}  {'Kept':>8}  {'Keep%':>7}")
    print("-" * 65)
    r = ablation_results.get("no_nms")
    if r:
        print(f"{'no_nms':>8}  {r['mean_AP']:>8.4f}  {r['mean_AP50']:>8.4f}  "
              f"{r['mean_AP75']:>8.4f}  {r['mean_AR_100']:>8.4f}  {r['total_kept']:>8}  {'100.0':>7}%")
    for thr in thresholds:
        r = ablation_results.get(thr)
        if r:
            print(f"{thr:>8.2f}  {r['mean_AP']:>8.4f}  {r['mean_AP50']:>8.4f}  "
                  f"{r['mean_AP75']:>8.4f}  {r['mean_AR_100']:>8.4f}  {r['total_kept']:>8}  {r['keep_ratio']:>6.1f}%")

    best_key = max(ablation_results, key=lambda k: ablation_results[k]["mean_AP"])
    best_thr_str = f"{best_key}" if best_key == "no_nms" else f"{best_key:.2f}"
    print(f"\n  Best NMS threshold = {best_thr_str}  (AP={ablation_results[best_key]['mean_AP']:.4f})")

    return ablation_results


def run_ablation_threshold(args, pkl_files, submission_dir):
    """Run score threshold ablation: filter predictions by minimum score."""
    subset_data = {}
    for pkl_file in pkl_files:
        subset = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(submission_dir, pkl_file)
        ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")
        if not os.path.isfile(ann_json):
            continue
        predictions = load_submission(pkl_path)
        if predictions:
            subset_data[subset] = {"predictions": predictions, "ann_json": ann_json}

    thresholds = [round(x * 0.05, 2) for x in range(0, 19)]  # 0.00, 0.05, ..., 0.90
    ablation_results = {}
    total_preds = sum(len(d["predictions"]) for d in subset_data.values())

    for thr in thresholds:
        all_metrics = {}
        total_kept = 0
        for subset, data in subset_data.items():
            filtered = [p for p in data["predictions"] if p["score"] >= thr]
            total_kept += len(filtered)
            m = evaluate_subset(data["ann_json"], filtered)
            if m:
                all_metrics[subset] = m
        if all_metrics:
            mean_ap = np.mean([m["AP"] for m in all_metrics.values()])
            mean_ap50 = np.mean([m["AP_50"] for m in all_metrics.values()])
            mean_ap75 = np.mean([m["AP_75"] for m in all_metrics.values()])
            mean_ar100 = np.mean([m["AR_100"] for m in all_metrics.values()])
            ablation_results[thr] = {
                "mean_AP": float(mean_ap),
                "mean_AP50": float(mean_ap50),
                "mean_AP75": float(mean_ap75),
                "mean_AR_100": float(mean_ar100),
                "num_subsets": len(all_metrics),
                "total_kept": total_kept,
                "keep_ratio": round(total_kept / total_preds * 100, 1) if total_preds > 0 else 0,
            }

    # Print table
    print(f"\n{'Thr':>6}  {'AP':>8}  {'AP50':>8}  {'AP75':>8}  {'AR100':>8}  {'Kept':>8}  {'Keep%':>7}")
    print("-" * 65)
    for thr in thresholds:
        r = ablation_results.get(thr)
        if r:
            print(f"{thr:>6.2f}  {r['mean_AP']:>8.4f}  {r['mean_AP50']:>8.4f}  "
                  f"{r['mean_AP75']:>8.4f}  {r['mean_AR_100']:>8.4f}  {r['total_kept']:>8}  {r['keep_ratio']:>6.1f}%")

    best_thr = max(ablation_results, key=lambda t: ablation_results[t]["mean_AP"])
    print(f"\n  Best threshold = {best_thr:.2f}  (AP={ablation_results[best_thr]['mean_AP']:.4f}, "
          f"kept={ablation_results[best_thr]['keep_ratio']:.1f}%)")

    return ablation_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FSOD submission")
    parser.add_argument("--submission", type=str, required=True,
                        help="Path to submission directory containing pkl files")
    parser.add_argument("--data-path", type=str, default="./data",
                        help="Path to RF20VL data directory with GT annotations")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "valid"],
                        help="Which split's annotations to evaluate against")
    parser.add_argument("--subset", type=str, default=None,
                        help="Evaluate only this subset (default: all)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save evaluation results as JSON")
    parser.add_argument("--ablation-topk", action="store_true",
                        help="Run top-K% ablation study (10%~100%, step 10%)")
    parser.add_argument("--ablation-threshold", action="store_true",
                        help="Run score threshold ablation study (0.05~0.90, step 0.05)")
    parser.add_argument("--ablation-nms", action="store_true",
                        help="Run NMS IoU threshold ablation study (0.50~0.90, step 0.05)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Discover pkl files in submission directory
    submission_dir = args.submission
    if not os.path.isdir(submission_dir):
        print(f"Error: submission directory not found: {submission_dir}", file=sys.stderr)
        sys.exit(1)

    pkl_files = sorted(f for f in os.listdir(submission_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"Error: no pkl files found in {submission_dir}", file=sys.stderr)
        sys.exit(1)

    if args.subset:
        pkl_files = [f for f in pkl_files if args.subset in f]
        if not pkl_files:
            print(f"Error: no pkl file matching subset '{args.subset}'", file=sys.stderr)
            sys.exit(1)

    print(f"Submission: {submission_dir}")
    print(f"Data path:  {args.data_path}")
    print(f"Split:      {args.split}")
    print(f"Subsets:    {len(pkl_files)}")
    print("=" * 90)

    if args.ablation_topk:
        ablation_results = run_ablation_topk(args, pkl_files, submission_dir)
        if args.output_json:
            output = {
                "submission_dir": submission_dir,
                "data_path": args.data_path,
                "split": args.split,
                "ablation_topk": {str(k): v for k, v in ablation_results.items()},
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output_json}")
        return

    if args.ablation_threshold:
        ablation_results = run_ablation_threshold(args, pkl_files, submission_dir)
        if args.output_json:
            output = {
                "submission_dir": submission_dir,
                "data_path": args.data_path,
                "split": args.split,
                "ablation_threshold": {str(k): v for k, v in ablation_results.items()},
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output_json}")
        return

    if args.ablation_nms:
        ablation_results = run_ablation_nms(args, pkl_files, submission_dir)
        if args.output_json:
            output = {
                "submission_dir": submission_dir,
                "data_path": args.data_path,
                "split": args.split,
                "ablation_nms": {str(k): v for k, v in ablation_results.items()},
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output_json}")
        return

    all_results = {}  # {subset: {"original": metrics, "ranking": metrics} or None}
    zero_metrics = {"AP": 0.0, "AP_50": 0.0, "AP_75": 0.0, "AP_small": 0.0,
                    "AP_medium": 0.0, "AP_large": 0.0, "AR_1": 0.0, "AR_10": 0.0, "AR_100": 0.0}

    for pkl_file in pkl_files:
        subset = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(submission_dir, pkl_file)
        ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")

        if not os.path.isfile(ann_json):
            print(f"  [{subset}] SKIP — annotation not found: {ann_json}")
            all_results[subset] = None
            continue

        predictions = load_submission(pkl_path)
        if not predictions:
            print(f"  [{subset}] SKIP — no predictions")
            all_results[subset] = None
            continue

        # Original evaluation
        metrics_orig = evaluate_subset(ann_json, predictions)
        # Ranking rescore evaluation
        predictions_ranked = apply_ranking_rescore(predictions)
        metrics_rank = evaluate_subset(ann_json, predictions_ranked)

        entry = {}
        if metrics_orig:
            entry["original"] = metrics_orig
        else:
            entry["original"] = zero_metrics
        if metrics_rank:
            entry["ranking"] = metrics_rank
        else:
            entry["ranking"] = zero_metrics
        all_results[subset] = entry

        o = metrics_orig or zero_metrics
        r = metrics_rank or zero_metrics
        print(f"  [{subset}] preds={len(predictions)}")
        print(f"    original: AP={o.get('AP', 0):.4f}  AP50={o.get('AP_50', 0):.4f}  "
              f"AP75={o.get('AP_75', 0):.4f}  "
              f"AR_1={o.get('AR_1', 0):.4f}  AR_10={o.get('AR_10', 0):.4f}  AR_100={o.get('AR_100', 0):.4f}")
        print(f"    ranking:  AP={r.get('AP', 0):.4f}  AP50={r.get('AP_50', 0):.4f}  "
              f"AP75={r.get('AP_75', 0):.4f}  "
              f"AR_1={r.get('AR_1', 0):.4f}  AR_10={r.get('AR_10', 0):.4f}  AR_100={r.get('AR_100', 0):.4f}")

    # Summary — use total pkl count as denominator so skipped subsets count as 0
    num_total = len(pkl_files)
    print("=" * 90)
    for eval_type in ["original", "ranking"]:
        metrics_list = [v[eval_type] for v in all_results.values() if v is not None and eval_type in v]
        num_skipped = sum(1 for v in all_results.values() if v is None)
        if not metrics_list:
            continue
        mean_ap = sum(m["AP"] for m in metrics_list) / num_total
        mean_ap50 = sum(m["AP_50"] for m in metrics_list) / num_total
        mean_ap75 = sum(m["AP_75"] for m in metrics_list) / num_total
        mean_ar1 = sum(m["AR_1"] for m in metrics_list) / num_total
        mean_ar10 = sum(m["AR_10"] for m in metrics_list) / num_total
        mean_ar100 = sum(m["AR_100"] for m in metrics_list) / num_total
        print(f"  [{eval_type}] Evaluated {len(metrics_list)}/{num_total} subsets"
              f"{f' ({num_skipped} skipped, counted as 0)' if num_skipped else ''}")
        print(f"  [{eval_type}] Mean AP    = {mean_ap:.4f}")
        print(f"  [{eval_type}] Mean AP50  = {mean_ap50:.4f}")
        print(f"  [{eval_type}] Mean AP75  = {mean_ap75:.4f}")
        print(f"  [{eval_type}] Mean AR_1  = {mean_ar1:.4f}")
        print(f"  [{eval_type}] Mean AR_10 = {mean_ar10:.4f}")
        print(f"  [{eval_type}] Mean AR_100= {mean_ar100:.4f}")

    # Save JSON
    if args.output_json:
        output = {
            "submission_dir": submission_dir,
            "data_path": args.data_path,
            "split": args.split,
            "num_subsets_total": num_total,
            "per_subset": {},
        }
        for eval_type in ["original", "ranking"]:
            metrics_list = [v[eval_type] for v in all_results.values() if v is not None and eval_type in v]
            if metrics_list:
                output[f"mean_AP_{eval_type}"] = float(sum(m["AP"] for m in metrics_list) / num_total)
                output[f"mean_AP50_{eval_type}"] = float(sum(m["AP_50"] for m in metrics_list) / num_total)
                output[f"mean_AP75_{eval_type}"] = float(sum(m["AP_75"] for m in metrics_list) / num_total)
                output[f"mean_AR_1_{eval_type}"] = float(sum(m["AR_1"] for m in metrics_list) / num_total)
                output[f"mean_AR_10_{eval_type}"] = float(sum(m["AR_10"] for m in metrics_list) / num_total)
                output[f"mean_AR_100_{eval_type}"] = float(sum(m["AR_100"] for m in metrics_list) / num_total)
            else:
                output[f"mean_AP_{eval_type}"] = None
        output["per_subset"] = {
            k: {ek: {kk: round(vv, 6) for kk, vv in ev.items()} for ek, ev in v.items()}
            for k, v in all_results.items() if v is not None
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
