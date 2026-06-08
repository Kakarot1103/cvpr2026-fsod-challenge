import os
import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
import torchvision
from torchvision.ops import box_iou
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
from sam3.sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.sam3.visualization_utils import normalize_bbox

# Default model paths — override via SAM3_MODEL_DIR / SAM3_CHECKPOINT env vars
_DEFAULT_MODEL_DIR = os.environ.get("SAM3_MODEL_DIR", "./pretrained/sam3")
_DEFAULT_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT", os.path.join(_DEFAULT_MODEL_DIR, "sam3.pt")
)


class Sam3Segmenter(nn.Module):
    def __init__(self, model_path: str = _DEFAULT_MODEL_DIR, device: str = None,
                 checkpoint_path: str = _DEFAULT_CHECKPOINT):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=checkpoint_path).to(self.device)
        self.processor = Sam3Processor(self.model)

    def _boxes_to_normalized_cxcywh(self, boxes_xyxy, width, height):
        if boxes_xyxy.numel() == 0:
            return []
        cxcywh = box_xyxy_to_cxcywh(boxes_xyxy.view(-1, 4))
        return normalize_bbox(cxcywh, width, height).tolist()

    def _get_boxes_from_state(self, inference_state):
        boxes = inference_state.get('boxes')
        if boxes is None or boxes.numel() == 0:
            return torch.zeros(0, 4, device=self.device)
        return boxes

    def _get_scores_from_state(self, inference_state):
        scores = inference_state.get('scores')
        if scores is None or scores.numel() == 0:
            return torch.zeros(0, device=self.device)
        return scores

    def get_query_boxes_from_cat(self, cat_img, cat_bboxes_xyxy, prompt=None):
        """
        从拼接图像（ref+query）中提取 query 半边的预测 boxes。

        Args:
            cat_img: 拼接后的PIL图像
            cat_bboxes_xyxy: 拼接图的 bbox (xyxy format tensor)
            prompt: 可选文本提示词，与 geometric prompt 联合使用
        Returns:
            query_boxes: query半边的预测 boxes (xyxy format tensor, 坐标已映射回 query 图空间)
        """
        width, height = cat_img.size
        inference_state = self.processor.set_image(cat_img)

        if cat_bboxes_xyxy.numel() > 0:
            norm_boxes_cxcywh = self._boxes_to_normalized_cxcywh(cat_bboxes_xyxy, width, height)
            for box in norm_boxes_cxcywh:
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state, box=box, label=True)

        if prompt and prompt != 'visual':
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        all_boxes = self._get_boxes_from_state(inference_state)

        if all_boxes.numel() > 0:
            centers_x = (all_boxes[:, 0] + all_boxes[:, 2]) / 2
            if height == 1024:
                mid_w, mid_h = width // 2, height // 2
                centers_y = (all_boxes[:, 1] + all_boxes[:, 3]) / 2
                keep = (centers_x > mid_w) & (centers_y > mid_h)
                query_boxes = all_boxes[keep].clone()
                query_boxes[:, 0] -= mid_w
                query_boxes[:, 2] -= mid_w
                query_boxes[:, 1] -= mid_h
                query_boxes[:, 3] -= mid_h
            else:
                mid_w = width // 2
                keep = centers_x > mid_w
                query_boxes = all_boxes[keep].clone()
                query_boxes[:, 0] -= mid_w
                query_boxes[:, 2] -= mid_w
        else:
            query_boxes = torch.zeros(0, 4, device=self.device)

        del inference_state
        return query_boxes

    def forward_with_query_boxes(self, query_img, query_bboxes_xyxy, prompt):
        """
        使用 query 图像和 bbox 生成三种检测结果。

        Args:
            query_img: query图像 (PIL Image)
            query_bboxes_xyxy: query的bbox (xyxy format tensor)
            prompt: 文本提示词
        Returns:
            dict: {
                'visual_boxes': tensor (N_v, 4),
                'tv_boxes': tensor (N_tv, 4),
                'text_boxes': tensor (N_t, 4),
                'text_presence_score': float
            }
        """
        width, height = query_img.size
        inference_state = self.processor.set_image(query_img)

        if query_bboxes_xyxy.numel() > 0:
            norm_boxes_cxcywh = self._boxes_to_normalized_cxcywh(query_bboxes_xyxy, width, height)
            for box in norm_boxes_cxcywh:
                if sum(box) == 0:
                    continue
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state, box=box, label=True)

        # ----------visual only-----------------
        visual_boxes = self._get_boxes_from_state(inference_state)
        visual_scores = self._get_scores_from_state(inference_state)

        # ----------visual+text / text only-----------------
        if prompt == 'visual':
            tv_boxes = visual_boxes
            tv_scores = visual_scores
            text_boxes = torch.zeros(0, 4, device=self.device)
            text_scores = torch.zeros(0, device=self.device)
            text_presence_score = 0.0
        else:
            # visual + text
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            tv_boxes = self._get_boxes_from_state(inference_state)
            tv_scores = self._get_scores_from_state(inference_state)

            # text only
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            text_boxes = self._get_boxes_from_state(inference_state)
            text_scores = self._get_scores_from_state(inference_state)
            text_presence_score = inference_state.get('presence_score', 0.0)

        return {
            'visual_boxes': visual_boxes,
            'visual_scores': visual_scores,
            'tv_boxes': tv_boxes,
            'tv_scores': tv_scores,
            'text_boxes': text_boxes,
            'text_scores': text_scores,
            'text_presence_score': text_presence_score,
        }

    def calculate_ref_iou_with_query_boxes(self, ref_img_pil, ref_bboxes_gt, query_img_pil, query_boxes):
        """
        使用 query boxes 作为 prompts 在 ref+query 拼接图上推理，
        提取 ref 半边的预测 boxes 与 gt 计算 box IoU。

        Args:
            ref_img_pil: ref 图像 (PIL Image)
            ref_bboxes_gt: ref 的 ground truth boxes (xyxy tensor)
            query_img_pil: query 图像 (PIL Image)
            query_boxes: query 的预测 boxes (xyxy tensor)
        Returns:
            float: 最佳 box IoU
        """
        if query_boxes.numel() == 0:
            return 0.0

        if not isinstance(ref_bboxes_gt, torch.Tensor):
            ref_bboxes_gt = torch.tensor(ref_bboxes_gt, dtype=torch.float32)
        else:
            ref_bboxes_gt = ref_bboxes_gt.clone().float()
        if ref_bboxes_gt.ndim == 1:
            ref_bboxes_gt = ref_bboxes_gt.unsqueeze(0)

        # 水平拼接图像
        ref_img_tensor = torch.from_numpy(np.array(ref_img_pil)).permute(2, 0, 1).float() / 255.0
        query_img_tensor = torch.from_numpy(np.array(query_img_pil)).permute(2, 0, 1).float() / 255.0
        cat_img = torch.cat([ref_img_tensor, query_img_tensor], dim=2)
        cat_img_pil = torchvision.transforms.functional.to_pil_image(cat_img)

        width, height = cat_img_pil.size
        ref_w = ref_img_pil.size[0]

        # 将 query boxes 平移到拼接图右半边
        shifted = query_boxes.clone()
        shifted[:, 0] += ref_w
        shifted[:, 2] += ref_w

        cat_inference_state = self.processor.set_image(cat_img_pil)
        norm_boxes = self._boxes_to_normalized_cxcywh(shifted, width, height)
        for box in norm_boxes:
            if sum(box) == 0:
                continue
            cat_inference_state = self.processor.add_geometric_prompt(
                state=cat_inference_state, box=box, label=True)

        all_boxes = self._get_boxes_from_state(cat_inference_state)
        best_iou = 0.0

        if all_boxes.numel() > 0:
            centers_x = (all_boxes[:, 0] + all_boxes[:, 2]) / 2
            ref_pred = all_boxes[centers_x < ref_w]
            if ref_pred.numel() > 0:
                ious = box_iou(ref_pred.cpu(), ref_bboxes_gt.cpu())
                if ious.numel() > 0:
                    best_iou = ious.max().item()

        del cat_inference_state
        return float(best_iou)

    def forward_with_cat(self, cat_img, cat_bboxes_xyxy, query_img, prompt):
        """
        便捷函数：组合调用 get_query_boxes_from_cat 和 forward_with_query_boxes。

        Args:
            cat_img: 拼接后的PIL图像
            cat_bboxes_xyxy: 拼接图的bbox (xyxy format tensor)
            query_img: query图像 (PIL Image)
            prompt: 文本提示词
        Returns:
            dict: 同 forward_with_query_boxes
        """
        query_boxes = self.get_query_boxes_from_cat(cat_img, cat_bboxes_xyxy)
        return self.forward_with_query_boxes(query_img, query_boxes, prompt)

    def select_prompt(self, ref_img_pil, ref_bboxes, candidates, query_img_pil,
                      score_weight: float = 0.5, alpha: float = 1.0, beta: float = 1.0):
        if not candidates:
            raise ValueError("select_prompt requires at least one candidate prompt.")

        if not isinstance(ref_bboxes, torch.Tensor):
            ref_bboxes = torch.tensor(ref_bboxes, dtype=torch.float32)
        else:
            ref_bboxes = ref_bboxes.clone().float()
        if ref_bboxes.ndim == 1:
            ref_bboxes = ref_bboxes.unsqueeze(0)

        best_score = float("-inf")
        best_prompt = candidates[0]
        candidate_scores = []

        ref_state = self.processor.set_image(ref_img_pil)
        query_state = self.processor.set_image(query_img_pil)

        for prompt in candidates:
            self.processor.reset_all_prompts(ref_state)
            self.processor.reset_all_prompts(query_state)

            ref_state = self.processor.set_text_prompt(state=ref_state, prompt=prompt)
            pred_boxes = self._get_boxes_from_state(ref_state)
            if pred_boxes.numel() > 0:
                ious = box_iou(pred_boxes.cpu(), ref_bboxes.cpu())
                ref_iou_value = ious.max().item() if ious.numel() > 0 else 0.0
            else:
                ref_iou_value = 0.0

            query_state = self.processor.set_text_prompt(state=query_state, prompt=prompt)
            query_score = query_state.get('presence_score', 0.0)
            query_score_value = float(query_score)

            if alpha == 0 and beta == 0:
                combined_score = random.random()
            else:
                combined_score = (ref_iou_value ** alpha) * (query_score_value ** beta)

            if combined_score > best_score:
                best_score = combined_score
                best_prompt = prompt

            candidate_scores.append({
                "prompt": prompt,
                "ref_iou": ref_iou_value,
                "query_score": query_score_value,
                "combined_score": float(combined_score),
            })

        if best_score < 1e-6:
            best_prompt = 'visual'
        return best_prompt, candidate_scores
