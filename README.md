# CVPR 2026 FSOD Challenge

CVPR 2026 Few-Shot Object Detection (FSOD) Challenge solution based on SAM3.

## Overview

This project tackles few-shot object detection on the **RF-20VL** dataset using the **SAM3** foundation model, combined with a Multimodal Large Language Model (MLLM) for post-hoc verification.

The pipeline consists of four stages:

1. **Candidate box generation via in-context learning**: Each support image is concatenated horizontally with the query image. SAM3 receives the support-side ground-truth boxes as geometric prompts and predicts target boxes on the query side through visual analogy.

2. **Text prompt optimization**: For each category, multiple candidate text prompts are evaluated on both support (reference IoU) and query (presence score) images to select the most effective one for SAM3's semantic space.

3. **Text-visual joint inference**: The optimized text prompt and the candidate boxes are jointly fed into SAM3, producing three types of predictions — visual-only, text+visual, and text-only — which are then merged and deduplicated via NMS.

4. **VQA rescoring**: A multimodal LLM (Qwen3-VL-8B) performs visual question answering on each detection. The query image is annotated with a red bounding box, and the model answers a Yes/No question about whether the target object lies inside the box. Normalized log-probabilities are used to rescore detection confidence, effectively reducing false positives.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+, PyTorch with CUDA
- SAM3 model (see below)

### Data

Place the RF-20VL dataset at `./data` (or create a symlink):

```bash
ln -s /path/to/RF-20VL ./data
```

### Pretrained Model

Download the SAM3 pretrained model and place it under `./pretrained/sam3/`:

```bash
mkdir -p pretrained/sam3
# Place sam3.pt in the directory
```

Or set the environment variable:

```bash
export SAM3_CHECKPOINT=/path/to/sam3.pt
```

## Usage

### Multi-GPU parallel inference

```bash
bash scripts/parallel/run_ddp.sh
```

### Evaluate a submission

```bash
python evaluate.py --submission results/<timestamp>/submissions/tv
python evaluate.py --submission results/<timestamp>/submissions/tv --split test --subset gwhd2021-fsod-atsv
```
