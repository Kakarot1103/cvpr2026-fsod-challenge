import math
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from openai import OpenAI


class VQARescorer(nn.Module):
    """
    VQA Rescorer: 用 VLM 对检测结果做二次置信度校准。

    通过向 VLM 展示带红框标注的图片，询问框内是否为指定类别，
    根据 Yes/No token 的 logprob 计算归一化置信度分数。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        server_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        bbox_color: str = "red",
        bbox_width: int = 3,
        use_crop: bool = False,
        crop_expand: float = 0.5,
    ):
        """
        Args:
            model_name: 部署在 vLLM 上的模型名称（需与 vLLM 启动时的 --served-model-name 一致）。
            server_url: vLLM OpenAI 兼容 API 地址。
            api_key: API key（vLLM 本地部署通常为 "EMPTY"）。
            bbox_color: 红框颜色。
            bbox_width: 红框线宽（像素）。
            use_crop: 是否裁剪 bbox 附近区域后发给 VLM（而非发整图）。
            crop_expand: 裁剪时框各边外扩比例（相对框宽高），默认 0.5 即各边外扩 50%。
        """
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(base_url=server_url, api_key=api_key)
        self.bbox_color = bbox_color
        self.bbox_width = bbox_width
        self.use_crop = use_crop
        self.crop_expand = crop_expand

    def _draw_bbox(self, image: Image.Image, bbox_xywh) -> Image.Image:
        """在图片上画红框，返回副本。"""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox_xywh
        draw.rectangle([x, y, x + w, y + h], outline=self.bbox_color, width=self.bbox_width)
        return img

    @staticmethod
    def _build_prompt(category_name: str, class_description: str = None) -> str:
        """构建 VQA 问题文本。"""
        if class_description:
            return (
                f"Given the '{category_name}' class defined as follows: {class_description}\n\n"
                f"Is the main subject or object being referred to as: '{category_name}' located inside the red bounding box "
                f"in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, "
                f"with no part outside, and it must be the only object present inside - no other objects should appear "
                f"within the box."
            )
        return (
            f"Is the main subject or object being referred to as: '{category_name}' located inside the red bounding box "
            f"in the image? Please answer Yes or No. Note: The object should be entirely inside the bounding box, "
            f"with no part outside, and it must be the only object present inside - no other objects should appear "
            f"within the box."
        )

    def _call_vlm(self, image: Image.Image, prompt: str) -> float:
        """
        调用 VLM，返回 Yes token 的归一化概率。

        使用 logprobs 提取 Yes/No token 概率，计算:
            score = p(Yes) / (p(Yes) + p(No))

        Returns:
            float: [0, 1] 的置信度分数，或 -1.0 表示 VQA 失败。
        """
        import base64
        from io import BytesIO

        # 缩放图片以匹配 DetPO 的 2880x1620 上限
        max_w, max_h = 2880, 1620
        w, h = image.size
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )

        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        yes_logprob = None
        no_logprob = None
        for token_lp in top_logprobs:
            token = token_lp.token.strip()
            if token == "Yes" and yes_logprob is None:
                yes_logprob = token_lp.logprob
            elif token == "yes" and yes_logprob is None:
                yes_logprob = token_lp.logprob
            elif token == "No" and no_logprob is None:
                no_logprob = token_lp.logprob
            elif token == "no" and no_logprob is None:
                no_logprob = token_lp.logprob

        # 两者都缺失 → VQA 失败
        if yes_logprob is None and no_logprob is None:
            return -1.0

        # Yes 缺失但 No 存在 → Yes 不在 top-20，概率极低
        if yes_logprob is None:
            return 0.0

        yes_prob = math.exp(yes_logprob)
        no_prob = math.exp(no_logprob) if no_logprob is not None else 0.0
        return yes_prob / (yes_prob + no_prob + 1e-18)

    def _crop_around_bbox(self, image: Image.Image, bbox_xywh) -> tuple:
        """裁剪 bbox 外扩 crop_expand 倍的区域，返回 (cropped_image, box_xywh_in_crop)。"""
        x, y, w, h = bbox_xywh
        pad_x = w * self.crop_expand
        pad_y = h * self.crop_expand

        img_w, img_h = image.size
        cx = max(0, int(x - pad_x))
        cy = max(0, int(y - pad_y))
        cx2 = min(img_w, int(x + w + pad_x))
        cy2 = min(img_h, int(y + h + pad_y))

        cropped = image.crop((cx, cy, cx2, cy2))
        box_in_crop = [x - cx, y - cy, w, h]
        return cropped, box_in_crop

    def forward(
        self,
        image: Image.Image,
        bbox_xywh,
        category_name: str,
        class_description: str = None,
    ) -> float:
        """
        对单个检测结果做 VQA 重打分。

        Args:
            image: 原始图片（PIL Image）。
            bbox_xywh: 检测框，格式 [x, y, w, h]。
            category_name: 类别名称。
            class_description: 可选的类别描述（来自数据集指令），提供后会增加上下文。

        Returns:
            vqa_score: Yes 概率归一化值，范围 (0, 1)。
        """
        if self.use_crop:
            cropped, box_in_crop = self._crop_around_bbox(image, bbox_xywh)
            img_with_bbox = self._draw_bbox(cropped, box_in_crop)
        else:
            img_with_bbox = self._draw_bbox(image, bbox_xywh)
        prompt = self._build_prompt(category_name, class_description)
        return self._call_vlm(img_with_bbox, prompt)
