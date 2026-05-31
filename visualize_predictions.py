"""
Visualize prediction results against GT annotations.

For each image, creates a folder named after the image (without extension).
Inside each folder, saves one image per predicted bbox showing:
  - All GT boxes in green
  - The single predicted bbox in red
  - Filename encodes max IoU with GT and confidence score: iou_XX.X_score_YY.Y.jpg
"""

import argparse
import os
import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm


def xywh_to_xyxy(box):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def draw_bbox(img, box, color, thickness=2, label=None):
    """Draw a [x, y, w, h] bbox on the image."""
    x1, y1, x2, y2 = [int(v) for v in xywh_to_xyxy(box)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs GT")
    parser.add_argument("--submission-dir", type=str,
                        default="submission/test/vqa-test-ddp-new-prompt/20260518_232312_vqa",
                        help="Directory containing per-subset pkl files")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root directory of RF-20VL dataset")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subsets", type=str, nargs="*", default=None,
                        help="Only process these subsets (default: all)")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Minimum score for predictions to visualize")
    args = parser.parse_args()

    # Collect subset pkl files
    pkl_files = sorted(f for f in os.listdir(args.submission_dir) if f.endswith(".pkl"))
    total_subsets = len(pkl_files)

    for pkl_idx, pkl_file in enumerate(pkl_files, 1):
        subset = pkl_file.replace(".pkl", "")
        if args.subsets and subset not in args.subsets:
            continue

        # Load predictions
        pkl_path = os.path.join(args.submission_dir, pkl_file)
        with open(pkl_path, "rb") as f:
            predictions = pickle.load(f)

        # Load GT annotations
        ann_path = os.path.join(args.data_dir, subset, args.split, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            print(f"[{pkl_idx}/{total_subsets}] {subset}: skipped (annotation not found)")
            continue

        with open(ann_path, "r") as f:
            coco = json.load(f)

        # Build lookups
        img_info = {img["id"]: img for img in coco["images"]}
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        gt_bboxes = {}  # (image_id, category_id) -> list of [x, y, w, h]
        for ann in coco["annotations"]:
            gt_bboxes.setdefault((ann["image_id"], ann["category_id"]), []).append(ann["bbox"])

        # Process each predicted sample
        num_saved = 0
        desc = f"[{pkl_idx}/{total_subsets}] {subset}"
        for sample in tqdm(predictions, desc=desc, unit="img"):
            image_id = sample["image_id"]
            instances = sample["instances"]

            if image_id not in img_info:
                continue

            img_meta = img_info[image_id]
            img_path = os.path.join(args.data_dir, subset, args.split, img_meta["file_name"])

            if not os.path.isfile(img_path):
                continue

            img_stem = os.path.splitext(img_meta["file_name"])[0]

            # Group predictions by category
            by_cat = {}
            for inst in instances:
                if inst["score"] < args.score_threshold:
                    continue
                by_cat.setdefault(inst["category_id"], []).append(inst)

            # Load image once
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            for cat_id, cat_instances in by_cat.items():
                cat_name = categories.get(cat_id, f"cat{cat_id}")
                sample_dir = os.path.join(args.output_dir, subset, cat_name, img_stem)
                os.makedirs(sample_dir, exist_ok=True)

                gts = gt_bboxes.get((image_id, cat_id), [])

                for inst in cat_instances:
                    pred_box = inst["bbox"]  # [x, y, w, h]
                    score = inst["score"]

                    # Compute max IoU with any GT box
                    pred_xyxy = xywh_to_xyxy(pred_box)
                    max_iou = 0.0
                    for gt_box in gts:
                        gt_xyxy = xywh_to_xyxy(gt_box)
                        iou = compute_iou(pred_xyxy, gt_xyxy)
                        max_iou = max(max_iou, iou)

                    # Draw on a fresh copy
                    vis = img_bgr.copy()

                    # Draw all GT boxes in green
                    for gt_box in gts:
                        draw_bbox(vis, gt_box, (0, 200, 0), thickness=2, label="GT")

                    # Draw this predicted bbox in red
                    iou_pct = max_iou * 100
                    draw_bbox(vis, pred_box, (0, 0, 220), thickness=2,
                              label=f"IoU:{iou_pct:.1f}% S:{score:.2f}")

                    # Save
                    fname = f"iou_{iou_pct:05.1f}_score_{score:.2f}.jpg"
                    cv2.imwrite(os.path.join(sample_dir, fname), vis)
                    num_saved += 1

        tqdm.write(f"{desc}: saved {num_saved} bbox images across {len(predictions)} query images")


if __name__ == "__main__":
    main()
