# CVPR 2026 Few-Shot Object Detection Challenge 技术报告

## 1. 任务描述

本方法面向 RF-20VL 数据集的 Few-Shot Object Detection (FSOD) 任务。RF-20VL 包含 20 个子集，涵盖农业、医学、工业检测、遥感等多个领域，每个子集提供少量带标注的训练图像（support set）和待检测的测试图像（query set），标注格式为 COCO JSON。

## 2. 方法概述

本方法基于 SAM3 (Segment Anything Model 3) 构建推理流水线。SAM3 是一个支持几何提示（bounding box）和文本提示（text prompt）的多模态分割模型，具备上下文学习能力。整个流水线分为四个阶段：

1. **文本提示词获取** — 为每个类别选取最优的文本提示词
2. **视觉提示框获取** — 利用 support 图像生成 query 图像上的候选检测框
3. **文本+视觉联合推理** — 将文本提示和候选框同时输入 SAM3，得到最终检测结果
4. **VQA 重排序** — 利用视觉语言模型对检测结果进行二次置信度校准

## 3. 方法细节

### 3.1 文本提示词获取

类别名称直接作为 SAM3 的文本提示时，可能因表述不精确而影响检测效果。例如 "Wheat Head" 可能不如 "wheat spike" 或 "grain head" 更贴合 SAM3 的语义空间。为此，本方法引入了提示词优化机制：

**候选提示词生成**: 为每个类别构造多个候选提示词（包括原始类别名称及其同义词、别名等变体）。

**提示词评分与选择** (`select_prompt`): 对每个候选提示词计算综合评分：

$$\text{score} = \text{ref\_iou}^\alpha \times \text{query\_score}^\beta$$

其中：
- **ref_iou（参考 IoU）**: 在 support 图像上使用该提示词进行纯文本推理，将预测框与 GT 框计算 IoU，评估提示词对已知样本的定位能力
- **query_score（query 置信度）**: 在 query 图像上使用该提示词，获取 SAM3 输出的 presence score，评估提示词对未知样本的语义激活程度

**预计算提示映射**: 为避免推理时的额外计算开销，预先搜索每个子集每个类别的最优提示词，保存为映射文件（`prompts/sam3_prompt_mapping/{subset}.json`）。推理时直接查表获取最优提示词，若映射表中无对应类别则回退到原始类别名称。

### 3.2 视觉提示框获取

本阶段利用训练集中的 support 图像生成 query 图像上的候选检测框，作为 SAM3 的几何提示输入。对每个 query 样本，遍历该类别的所有 support 图像：

**步骤 1 — 图像拼接**: 将 support 图像（左）和 query 图像（右）水平拼接。若两者高度不同，将 support 图像按比例缩放至与 query 等高，保持宽高比不变。

**步骤 2 — 几何提示注入**: 将 support 图像中的 GT 边界框（经缩放校正后转为 xyxy 格式）归一化为 cxcywh 格式，作为几何提示输入 SAM3，对拼接图进行推理。

**步骤 3 — Query 侧候选框提取**: SAM3 在拼接图上输出预测框，根据框中心点 x 坐标是否落在右半侧（即 query 侧）进行筛选。将筛选后的框坐标向左偏移（减去 support 图像宽度），映射回 query 图像空间，并将坐标裁剪至图像边界内。

**步骤 4 — 多 Support 融合与 NMS**: 收集所有 support 图像产生的候选框，拼接后使用 NMS（IoU 阈值 0.5）去重，得到最终的候选框集合。

> 核心思想：SAM3 具备上下文理解能力，当拼接图左侧标注了目标位置时，模型能通过视觉类比在右侧 query 侧生成对应位置的目标候选框。

### 3.3 文本+视觉联合推理

将阶段 3.1 获取的文本提示词和阶段 3.2 获取的候选框同时输入 SAM3，进行联合推理（`forward_with_query_boxes`）：

1. **图像编码**: 对 query 图像进行 SAM3 的图像编码
2. **几何提示注入**: 将候选框（归一化为 cxcywh 格式）逐一作为 positive geometric prompt 注入
3. **文本提示注入**: 设置文本提示词（来自阶段 3.1 的最优提示词）
4. **预测输出**: SAM3 综合利用空间位置信息和语义信息，输出最终的检测框及对应置信度分数
5. **NMS 后处理**: 对预测框使用 NMS（IoU 阈值 0.5）去除冗余检测

### 3.4 VQA 重排序

利用视觉语言模型（VLM）对阶段 3.3 的 TV 预测结果进行二次置信度校准，提升检测精度：

**输入构建**: 在 query 图像上用红框标注每个检测框的位置。

**VQA 提问**: 向 VLM 提出 Yes/No 二元问题：

> "Is the main subject or object being referred to as: '{category_name}' located inside the red bounding box in the image? Please answer Yes or No."

同时可附加类别描述（来自 DetPO 生成的细粒度类描述），为 VLM 提供更丰富的语义上下文。例如对 "Wheat Head" 类别，会附带其在数据集中的详细外观描述。

**分数计算**: 从 VLM 输出的 logprobs 中提取 `Yes` 和 `No` token 的对数概率，计算归一化置信度：

$$\text{vqa\_score} = \frac{p(\text{Yes})}{p(\text{Yes}) + p(\text{No})}$$

**回退策略**: 当 VQA 调用失败时（如 Yes 和 No 均不在 top-20 tokens 中），保留原始 TV 模型的预测分数。

**VLM 配置**: 使用 Qwen3-VL-8B-Instruct，通过 vLLM 本地部署为 OpenAI 兼容 API（`max_tokens=1`, `temperature=0.0`, `logprobs=True`）。

## 4. 推理配置

| 参数 | 值 |
|------|------|
| 模型 | SAM3 (sam3.pt) |
| VQA 模型 | Qwen3-VL-8B-Instruct (vLLM 部署) |
| GPU 数量 | 5 |
| 候选框 NMS IoU 阈值 | 0.5 |
| 预测框 NMS IoU 阈值 | 0.5 |
| VQA 类别描述来源 | DetPO/prompts/detpo/Qwen3-VL-8B-Instruct |
| 提示词映射来源 | prompts/sam3_prompt_mapping |
| VQA 重排序目标 | tv 预测结果 |

## 5. 流水线图

```
Input: (query_image, category) + all support images & bboxes
                          │
          ┌───────────────┼────────────────┐
          ▼               │                ▼
  ┌──────────────┐        │        ┌──────────────┐
  │ Step 1:      │        │        │ Step 2:      │
  │ 文本提示词获取 │        │        │ 视觉提示框获取 │
  │              │        │        │              │
  │ prompt_mapping│       │        │ For each     │
  │ 查表获取最优  │        │        │ support:     │
  │ 类别提示词    │        │        │  concat(s,q) │
  │              │        │        │  SAM3 → boxes │
  └──────┬───────┘        │        │  filter右半侧 │
         │                │        │ Merge + NMS   │
         │  text_prompt   │        └───────┬───────┘
         │                │                │ candidate_boxes
         └────────┬───────┘────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Step 3:        │
         │ 文本+视觉联合推理 │
         │                │
         │ SAM3(query_img, │
         │   text_prompt,  │
         │   candidate_boxes)
         │ → tv_boxes     │
         │ + NMS (IoU=0.5)│
         └───────┬────────┘
                 │ tv predictions
                 ▼
         ┌────────────────┐
         │ Step 4:        │
         │ VQA 重排序      │
         │                │
         │ Qwen3-VL-8B:  │
         │ query+红框 →   │
         │ "Yes/No?" →    │
         │ logprobs →     │
         │ rescored_conf  │
         └───────┬────────┘
                 │
                 ▼
         Output: COCO 格式提交文件
```

## 6. 评估

使用 COCO 标准评估指标：AP、AP@0.5、AP@0.75、AP (small/medium/large)、AR@1/10/100。提交格式为 pkl 文件，每个子集一个文件，包含 `[{"image_id": int, "instances": [...]}]` 格式的检测结果。

## 7. 总结

本方法的核心贡献：

1. **视觉上下文驱动的候选框生成**: 利用 SAM3 的上下文学习能力，通过 support-query 图像拼接实现零参数的目标候选区域生成，无需额外训练。
2. **文本提示词自动优化**: 通过联合评估参考 IoU 和 query 置信度，自动为每个类别选取最适配 SAM3 语义空间的提示词。
3. **VLM 辅助的置信度校准**: 利用 VQA 机制对检测框进行二次验证，通过 logprobs 获取细粒度置信度分数，有效降低误检率。
