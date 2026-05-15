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

    all_results = {}
    all_ap50 = []

    for pkl_file in pkl_files:
        subset = os.path.splitext(pkl_file)[0]
        pkl_path = os.path.join(submission_dir, pkl_file)
        ann_json = os.path.join(args.data_path, subset, args.split, "_annotations.coco.json")

        if not os.path.isfile(ann_json):
            print(f"  [{subset}] SKIP — annotation not found: {ann_json}")
            continue

        predictions = load_submission(pkl_path)
        metrics = evaluate_subset(ann_json, predictions)

        if metrics:
            all_results[subset] = metrics
            all_ap50.append(metrics["AP_50"])
            print(f"  [{subset}] AP={metrics['AP']:.4f}  AP50={metrics['AP_50']:.4f}  "
                  f"AP75={metrics['AP_75']:.4f}  AR100={metrics['AR_100']:.4f}  "
                  f"preds={len(predictions)}")
        else:
            print(f"  [{subset}] SKIP — no predictions")

    # Summary
    print("=" * 90)
    if all_ap50:
        mean_ap = np.mean([m["AP"] for m in all_results.values()])
        mean_ap50 = np.mean(all_ap50)
        mean_ap75 = np.mean([m["AP_75"] for m in all_results.values()])
        print(f"  Evaluated {len(all_results)} subsets")
        print(f"  Mean AP   = {mean_ap:.4f}")
        print(f"  Mean AP50 = {mean_ap50:.4f}")
        print(f"  Mean AP75 = {mean_ap75:.4f}")
    else:
        print("  No results to summarize.")

    # Save JSON
    if args.output_json:
        output = {
            "submission_dir": submission_dir,
            "data_path": args.data_path,
            "split": args.split,
            "num_subsets_evaluated": len(all_results),
            "mean_AP": float(mean_ap) if all_ap50 else None,
            "mean_AP50": float(mean_ap50) if all_ap50 else None,
            "mean_AP75": float(mean_ap75) if all_ap50 else None,
            "per_subset": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in all_results.items()},
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
