# 任务：基于 DetPO 长 Prompt 生成 SAM3 优化类别名映射

## 目标

本脚本只负责生成并保存一个类别名映射文件


也就是：

```text
原始 category name → 优化后的 SAM3 短 prompt
```

注意：

- 本脚本 **不负责完整 query/test 推理**
- 本脚本 **不负责最终 VQA rescore**
- 完整推理会在另一个脚本中读取本脚本输出的 mapping 文件后执行

---

## 1. 读取 DetPO Prompt 文件

读取 DetPO 仓库中已经生成好的类别长描述文件，例如：

```text
DetPO/prompts/detpo/Qwen3-VL-8B-Instruct/all_refined_class_instructions_actions.json
```

解析出每个类别：

```python
category_name
detpo_long_prompt
```

输入格式示例：

```json
{
  "soft plastic": "Soft plastic refers to flexible, transparent or semi-transparent plastic material...",
  "hard plastic": "Hard plastic refers to rigid plastic objects..."
}
```

---

## 2. 为每个类别生成 n 个候选 SAM3 Prompt

对每个类别调用 LLM (LLM配置可参考inference_ddp.py)。

输入给 LLM：

```text
1. 原始 category name
2. DetPO long prompt
3. SAM3 prompt 规则
```

`n` 做成参数，不要写死：

```python
num_candidate_prompts = n
```

默认值可以设为：

```python
num_candidate_prompts = 5
```

SAM3 prompt 规则：

```text
SAM3 works best with short English noun phrases.
The prompt should be 1 to 3 words.
Do not output a sentence.
Do not include explanations.
Avoid overly broad words such as object, item, thing, region, area, target.
The prompt should be a concise visual category name suitable for object detection.
```

LLM 输出格式必须是 JSON：

```json
{
  "category": "soft plastic",
  "candidate_prompts": [
    "soft plastic",
    "plastic bag",
    "plastic wrapper",
    "plastic sheet",
    "plastic film"
  ]
}
```

要求：

- `candidate_prompts` 数量尽量等于 `n`
- 每个 prompt 必须是 1 到 3 个英文词
- 每个 prompt 必须是名词短语
- 不要输出解释性句子

---


## 4. 在 Support 上评估每个候选 Prompt

对每个类别的每个候选 prompt，在 training/support 图像上运行 SAM3。

这里的目的不是保存完整推理结果，而是为了选择最优 prompt。

对每个候选 prompt，计算它在该类别 support 上的表现。


已有 COCO eval 工具，可以计算AP：

最终每个类别选择得分最高的候选 prompt。

---


## 6. 保存候选 Prompt 和评估结果

保存位置

```text
prompts/sam3_prompt_mapping/actions-zzid2-zb1hq-fsod-amih.json
```

格式：

```json
{
  "soft plastic": {
    "soft plastic": {
      "recall_at_50": 0.45,
      "mean_best_iou": 0.39,
      "fp_per_image": 3.2,
      "prompt_score": 0.41
    },
    "plastic bag": {
      "recall_at_50": 0.61,
      "mean_best_iou": 0.48,
      "fp_per_image": 2.4,
      "prompt_score": 0.55
    }
  }
}
```
---

## 7. 需要支持的参数

建议支持以下命令行参数：

```text
--output-dir
--num-candidate-prompts
--subset
```

