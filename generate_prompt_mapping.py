"""
基于 DetPO 长 Prompt 生成 SAM3 优化类别名映射。

流程：
  1. 读取 DetPO 长描述文件
  2. 用 LLM 为每个类别生成 n 个候选短 prompt
  3. 在 train 图像上用 SAM3 text-only 模式评估每个候选 prompt（COCO AP）
  4. 保存映射文件到 prompts/sam3_prompt_mapping/{subset}.json
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from openai import OpenAI
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou

from datasets.rf20vl import get_all_subsets
from model.sam3 import Sam3Segmenter
from sam3.sam3.model.box_ops import box_xywh_to_xyxy, box_xyxy_to_xywh


# ---------------------------------------------------------------------------
# DetPO prompt 解析
# ---------------------------------------------------------------------------

def load_class_descriptions(prompt_dir, subset):
    """最长前缀匹配找到 DetPO prompt 文件并加载。"""
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


def parse_detpo_description(raw_value):
    """解析 DetPO 的描述值，支持两种格式：简单字符串或字符串化的 list-of-dict。"""
    if not isinstance(raw_value, str):
        return str(raw_value)
    stripped = raw_value.strip()
    if stripped.startswith("[{"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and len(parsed) > 0:
                first = parsed[0]
                if isinstance(first, dict):
                    return list(first.values())[0]
        except (ValueError, SyntaxError):
            pass
    if stripped == "[]":
        return ""
    return stripped


# ---------------------------------------------------------------------------
# LLM 候选 prompt 生成
# ---------------------------------------------------------------------------

SAM3_PROMPT_RULES = """SAM3 works best with short English noun phrases.
The prompt should be 1 to 3 words.
Do not output a sentence.
Do not include explanations.
Avoid overly broad words such as object, item, thing, region, area, target.
The prompt should be a concise visual category name suitable for object detection."""

SYSTEM_PROMPT = f"""You are an expert at creating concise visual category prompts for the SAM3 object detection model.

Rules for SAM3 prompts:
{SAM3_PROMPT_RULES}

You must output valid JSON with exactly this format:
{{"category": "<category_name>", "candidate_prompts": ["<prompt1>", "<prompt2>", ...]}}

Requirements:
- Generate exactly the requested number of candidate prompts
- Each prompt must be 1 to 3 English words
- Each prompt must be a noun phrase
- Do not include explanations or sentences
- Ensure diversity among candidates (different angles/aspects of the object)"""


def generate_candidate_prompts(client, model_name, category_name, detpo_description,
                               num_candidates=5, max_retries=3):
    """用 LLM 为单个类别生成候选 prompt 列表。"""
    user_msg = f"Category name: {category_name}\n"
    if detpo_description:
        desc = detpo_description[:2000]
        user_msg += f"Description: {desc}\n"
    user_msg += f"\nGenerate {num_candidates} candidate SAM3 prompts."

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            # 提取 JSON 块
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            candidates = data.get("candidate_prompts", [])
            # 过滤：1-3 个词的英文短语
            valid = []
            for c in candidates:
                c = c.strip().strip('"').strip("'")
                words = c.split()
                if 1 <= len(words) <= 3 and all(w.isalpha() or w in ("-/",) for w in words):
                    valid.append(c)
            if not valid:
                continue
            # 确保原始类别名在候选中
            if category_name not in valid:
                valid.insert(0, category_name)
            return valid[:num_candidates + 1]
        except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
            logging.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
            continue

    # fallback：只用原始类别名
    return [category_name]


# ---------------------------------------------------------------------------
# COCO 评估
# ---------------------------------------------------------------------------

def evaluate_subset(ann_json, predictions):
    """运行 COCO eval，返回指标字典。"""
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
    }


# ---------------------------------------------------------------------------
# Support 图像加载
# ---------------------------------------------------------------------------

def load_support_data(data_path, subset, max_images=None):
    """加载 train split 标注，按类别分组返回 support 数据。

    Returns:
        categories: {cat_id: cat_name}
        support_by_cat: {cat_id: [(img_id, img_path, [bbox_xywh, ...])]}
        ann_json_path: 标注文件路径
    """
    ann_json_path = os.path.join(data_path, subset, "train", "_annotations.coco.json")
    if not os.path.isfile(ann_json_path):
        return None, None, None

    with open(ann_json_path, "r") as f:
        ann_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in ann_data["categories"]}
    img_lookup = {img["id"]: img for img in ann_data["images"]}

    # 按类别分组
    bbox_by_cat_img = defaultdict(lambda: defaultdict(list))
    for ann in ann_data["annotations"]:
        bbox_by_cat_img[ann["category_id"]][ann["image_id"]].append(ann["bbox"])

    support_by_cat = {}
    subset_dir = os.path.join(data_path, subset, "train")
    for cat_id, img_dict in bbox_by_cat_img.items():
        items = []
        for img_id in sorted(img_dict.keys()):
            img_info = img_lookup[img_id]
            img_path = os.path.join(subset_dir, img_info["file_name"])
            items.append((img_id, img_path, img_dict[img_id]))
        if max_images is not None and max_images > 0:
            items = items[:max_images]
        support_by_cat[cat_id] = items

    return categories, support_by_cat, ann_json_path


# ---------------------------------------------------------------------------
# SAM3 候选 prompt 评估
# ---------------------------------------------------------------------------

def evaluate_prompts_for_category(segmenter, candidates, support_data, cat_id, ann_json_path):
    """对单个类别的所有候选 prompt 在 support 图像上做 text-only 评估。

    对每张 support 图像只编码一次 backbone，然后循环所有候选 prompt。
    收集所有候选的 predictions，统一跑一次 COCO eval。

    Returns:
        {prompt: {"AP": ..., "AP_50": ..., "AP_75": ..., "prompt_score": ...}}
    """
    # 为每个候选收集 predictions
    prompt_predictions = {p: [] for p in candidates}

    for img_id, img_path, gt_bboxes_xywh in support_data:
        pil_img = Image.open(img_path).convert("RGB")
        width, height = pil_img.size

        state = segmenter.processor.set_image(pil_img)

        for prompt in candidates:
            segmenter.processor.reset_all_prompts(state)
            state = segmenter.processor.set_text_prompt(state=state, prompt=prompt)

            pred_boxes = state.get("boxes")
            pred_scores = state.get("scores")

            if pred_boxes is not None and pred_boxes.numel() > 0:
                # xyxy -> xywh for COCO format
                pred_xywh = box_xyxy_to_xywh(pred_boxes.cpu())
                for i in range(pred_xywh.shape[0]):
                    prompt_predictions[prompt].append({
                        "image_id": int(img_id),
                        "category_id": int(cat_id),
                        "bbox": pred_xywh[i].tolist(),
                        "score": float(pred_scores[i].cpu()) if pred_scores is not None else 1.0,
                    })

        del state

    # 对每个候选跑 COCO eval
    results = {}
    for prompt in candidates:
        preds = prompt_predictions[prompt]
        metrics = evaluate_subset(ann_json_path, preds)
        if metrics:
            results[prompt] = {
                "AP": metrics["AP"],
                "AP_50": metrics["AP_50"],
                "AP_75": metrics["AP_75"],
                "prompt_score": metrics["AP_50"],
            }
        else:
            results[prompt] = {
                "AP": 0.0,
                "AP_50": 0.0,
                "AP_75": 0.0,
                "prompt_score": 0.0,
            }

    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_subset(subset, args, segmenter, llm_client):
    """处理单个 subset，返回映射结果。"""
    logging.info(f"Processing subset: {subset}")

    # 1. 加载 DetPO prompts
    raw_descriptions = load_class_descriptions(args.prompt_dir, subset)
    logging.info(f"  Loaded {len(raw_descriptions)} DetPO descriptions")

    # 2. 加载 support 数据
    categories, support_by_cat, ann_json_path = load_support_data(
        args.data_path, subset, max_images=args.max_support_images
    )
    if categories is None:
        logging.warning(f"  No train annotations found, skipping")
        return None

    # 3. 对每个类别处理
    all_results = {}
    total_cats = len(categories)
    for idx, (cat_id, cat_name) in enumerate(categories.items()):
        logging.info(f"  [{idx + 1}/{total_cats}] Category: {cat_name}")

        # 获取 support 数据
        support_data = support_by_cat.get(cat_id, [])
        if not support_data:
            logging.info(f"    No support images, skipping")
            continue

        # 获取 DetPO 描述
        raw_desc = raw_descriptions.get(cat_name, "")
        detpo_desc = parse_detpo_description(raw_desc) if raw_desc else ""

        # 生成候选 prompt
        candidates = generate_candidate_prompts(
            llm_client, args.llm_model, cat_name, detpo_desc,
            num_candidates=args.num_candidate_prompts,
        )
        logging.info(f"    Candidates: {candidates}")

        # 评估
        cat_results = evaluate_prompts_for_category(
            segmenter, candidates, support_data, cat_id, ann_json_path
        )
        all_results[cat_name] = cat_results

        # 打印每个候选的得分
        for p, m in cat_results.items():
            logging.info(f"      {p!r}: AP_50={m['AP_50']:.4f}")

        # 增量保存
        output_path = os.path.join(args.output_dir, f"{subset}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM3-optimized prompt mapping from DetPO descriptions")
    parser.add_argument("--data-path", default="./data", help="Dataset root directory")
    parser.add_argument("--subset", default=None, help="Process single subset (default: all)")
    parser.add_argument("--prompt-dir", default="DetPO/prompts/detpo/Qwen3-VL-8B-Instruct",
                        help="DetPO prompt directory")
    parser.add_argument("--output-dir", default="prompts/sam3_prompt_mapping",
                        help="Output directory for mapping files")
    parser.add_argument("--num-candidate-prompts", type=int, default=5,
                        help="Number of candidate prompts per category")
    parser.add_argument("--llm-server-url", default="http://localhost:22002/v1",
                        help="LLM server URL (OpenAI-compatible)")
    parser.add_argument("--llm-model", default="qwen3-vl-8b",
                        help="LLM model name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-support-images", type=int, default=None,
                        help="Limit support images per category")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # 日志
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 SAM3
    logging.info("Loading SAM3 model...")
    segmenter = Sam3Segmenter(device=args.device)

    # 初始化 LLM client
    llm_client = OpenAI(base_url=args.llm_server_url, api_key="EMPTY")

    # 确定要处理的 subset 列表
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = get_all_subsets(args.data_path)
    logging.info(f"Found {len(subsets)} subsets to process")

    for subset in subsets:
        result = process_subset(subset, args, segmenter, llm_client)
        if result:
            output_path = os.path.join(args.output_dir, f"{subset}.json")
            logging.info(f"  Saved to {output_path} ({len(result)} categories)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
