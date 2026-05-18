"""
将 DetPO 的 predictions JSON 转换为标准提交 pkl 格式。

用法:
    python scripts/detpo_to_submission.py \
        --json DetPO/results/test_aerial_airport/predictions/default/predictions_aerial-airport-7ap9o-fsod-ddgc_model.json \
        --subset aerial-airport-7ap9o-fsod-ddgc \
        --split test

    # 批量转换所有 DetPO 结果
    python scripts/detpo_to_submission.py --detpo-dir DetPO/results/run_all_20260517_070158 --split test
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime


def json_to_submission(predictions_json):
    """将 COCO predictions 列表转为按 image_id 分组的提交格式。"""
    grouped = defaultdict(list)
    for pred in predictions_json:
        grouped[pred["image_id"]].append(
            {
                "image_id": pred["image_id"],
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "score": pred["score"],
                "segmentation": [],
                "area": pred["bbox"][2] * pred["bbox"][3],
                "iscrowd": 0,
            }
        )

    submission = []
    # 为每个 instance 分配递增 id
    instance_id = 1
    for image_id in sorted(grouped.keys()):
        instances = grouped[image_id]
        for inst in instances:
            inst["id"] = instance_id
            instance_id += 1
        submission.append({"image_id": image_id, "instances": instances})
    return submission


def convert_single(json_path, output_dir, subset, split):
    """转换单个 JSON 文件。"""
    with open(json_path) as f:
        preds = json.load(f)

    submission = json_to_submission(preds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    pkl_path = os.path.join(output_dir, f"{subset}.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(submission, f)

    print(f"[OK] {len(submission)} images -> {pkl_path}")
    return pkl_path


def convert_batch(detpo_dir, split, output_base="submission/detpo"):
    """批量转换 DetPO 结果目录下所有 subset。"""
    pred_dir = os.path.join(detpo_dir, "predictions", "default")
    if not os.path.isdir(pred_dir):
        print(f"预测目录不存在: {pred_dir}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, f"{timestamp}")

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith("_model.json"):
            continue
        # 从文件名提取 subset: predictions_{subset}_model.json
        subset = fname.replace("predictions_", "").replace("_model.json", "")
        json_path = os.path.join(pred_dir, fname)
        convert_single(json_path, output_dir, subset, split)

    print(f"\n全部转换完成，输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DetPO predictions -> submission pkl")
    parser.add_argument("--json", type=str, help="单个 predictions JSON 文件路径")
    parser.add_argument("--subset", type=str, help="subset 名称（单文件模式）")
    parser.add_argument("--split", type=str, default="test", help="数据 split")
    parser.add_argument(
        "--detpo-dir",
        type=str,
        help="DetPO 结果根目录（批量模式，自动扫描 predictions/default/）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（单文件模式，默认 submission/detpo/{timestamp}）",
    )
    args = parser.parse_args()

    if args.detpo_dir:
        convert_batch(args.detpo_dir, args.split)
    elif args.json and args.subset:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"submission/detpo/{timestamp}"
        convert_single(args.json, output_dir, args.subset, args.split)
    else:
        parser.error("请指定 --detpo-dir（批量模式）或 --json + --subset（单文件模式）")


if __name__ == "__main__":
    main()
