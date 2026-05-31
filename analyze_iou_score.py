"""
Analyze IoU vs Score correlation across predictions.

Outputs:
  1. Per-subset scatter plot (IoU vs Score)
  2. Overall scatter plot
  3. Pearson & Spearman correlation coefficients
  4. Binned statistics (mean score per IoU range)
"""

import argparse
import os
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def collect_pairs(submission_dir, data_dir, split, score_threshold):
    """Collect (iou, score) pairs for all subsets."""
    pkl_files = sorted(f for f in os.listdir(submission_dir) if f.endswith(".pkl"))
    all_ious, all_scores, all_subsets = [], [], []

    for pkl_file in pkl_files:
        subset = pkl_file.replace(".pkl", "")

        with open(os.path.join(submission_dir, pkl_file), "rb") as f:
            predictions = pickle.load(f)

        ann_path = os.path.join(data_dir, subset, split, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path, "r") as f:
            coco = json.load(f)

        gt_bboxes = {}
        for ann in coco["annotations"]:
            gt_bboxes.setdefault((ann["image_id"], ann["category_id"]), []).append(
                [ann["bbox"][0], ann["bbox"][1],
                 ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
            )

        subset_ious, subset_scores = [], []
        for sample in predictions:
            image_id = sample["image_id"]
            for inst in sample["instances"]:
                if inst["score"] < score_threshold:
                    continue
                pred_xyxy = xywh_to_xyxy(inst["bbox"])
                gts = gt_bboxes.get((image_id, inst["category_id"]), [])
                max_iou = max((compute_iou(pred_xyxy, g) for g in gts), default=0.0)
                subset_ious.append(max_iou)
                subset_scores.append(inst["score"])

        if subset_ious:
            all_ious.append(np.array(subset_ious))
            all_scores.append(np.array(subset_scores))
            all_subsets.append(subset)

    return all_ious, all_scores, all_subsets


def plot_scatter(ious, scores, title, save_path, n_bins=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Scatter + correlation ---
    ax = axes[0]
    ax.scatter(scores, ious, s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Score")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    pr, _ = pearsonr(scores, ious)
    sr, _ = spearmanr(scores, ious)
    ax.text(0.02, 0.98, f"Pearson: {pr:.3f}\nSpearman: {sr:.3f}\nN={len(ious)}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # --- Binned mean ---
    ax = axes[1]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(ious, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    means = np.array([scores[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                      for i in range(n_bins)])
    counts = np.array([(bin_idx == i).sum() for i in range(n_bins)])
    centers = (bins[:-1] + bins[1:]) / 2

    valid = ~np.isnan(means)
    ax.bar(centers[valid], means[valid], width=bins[1] - bins[0], align="center",
           color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    for c, m, cnt in zip(centers[valid], means[valid], counts[valid]):
        ax.text(c, m + 0.01, str(cnt), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("IoU bin")
    ax.set_ylabel("Mean Score")
    ax.set_title("Mean Score per IoU bin")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return pr, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=str,
                        default="submission/test/vqa-test-ddp-new-prompt/20260518_232312_vqa")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="analysis_iou_score")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_ious, all_scores, all_subsets = collect_pairs(
        args.submission_dir, args.data_dir, args.split, args.score_threshold
    )

    # Per-subset plots
    for ious, scores, subset in zip(all_ious, all_scores, all_subsets):
        pr, sr = plot_scatter(
            ious, scores, subset,
            os.path.join(args.output_dir, f"{subset}.png")
        )
        print(f"  {subset}: Pearson={pr:.3f}, Spearman={sr:.3f}, N={len(ious)}")

    # Overall
    cat_ious = np.concatenate(all_ious)
    cat_scores = np.concatenate(all_scores)
    pr, sr = plot_scatter(
        cat_ious, cat_scores, "ALL subsets",
        os.path.join(args.output_dir, "_overall.png")
    )
    print(f"\nOverall: Pearson={pr:.3f}, Spearman={sr:.3f}, N={len(cat_ious)}")


if __name__ == "__main__":
    main()
