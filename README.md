# CVPR 2026 FSOD Challenge

CVPR 2026 Few-Shot Object Detection (FSOD) Challenge solution based on SAM3.

## Overview

This project tackles few-shot object detection on the **RF-20VL** dataset using the **SAM3** foundation model, combined with a Multimodal Large Language Model (MLLM) for post-hoc verification.

The pipeline consists of four stages:

1. **Text prompt optimization (pre-computed)**: Raw category names may not align well with SAM3's semantic space. The script `generate_prompt_mapping.py` first uses an LLM (e.g., Qwen3-VL-8B) to generate multiple candidate short prompts per category, then evaluates each candidate via SAM3 text-only inference on training images. The prompt achieving the highest COCO AP is selected and saved to `prompts/sam3_prompt_mapping/<subset>.json`. At inference time, `inference_ddp.py` simply loads this mapping and looks up the optimal prompt per category.

2. **Candidate box generation via in-context learning**: Each support image is concatenated horizontally with the query image. SAM3 receives the support-side ground-truth boxes as geometric prompts and predicts target boxes on the query side through visual analogy.

3. **Text-visual joint inference**: The optimized text prompt and the candidate boxes are jointly fed into SAM3, producing three types of predictions — visual-only, text+visual, and text-only — which are then merged and deduplicated via NMS.

4. **VQA rescoring (optional)**: A multimodal LLM (Qwen3-VL-8B) performs visual question answering on each detection. The query image is annotated with a red bounding box, and the model answers a Yes/No question about whether the target object lies inside the box. Normalized log-probabilities are used to rescore detection confidence, effectively reducing false positives.

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

### 1. Generate prompt mapping (pre-compute, requires a running LLM server)

```bash
python generate_prompt_mapping.py --subset <subset-name>
```

This connects to a local vLLM server (`http://localhost:22002/v1` by default) and saves per-subset mappings to `prompts/sam3_prompt_mapping/`. Pre-computed mappings are already included in this repo.

### 2. Multi-GPU parallel inference

```bash
bash scripts/parallel/run_ddp.sh
```

### Evaluate a submission

```bash
python evaluate.py --submission results/<timestamp>/submissions/tv
python evaluate.py --submission results/<timestamp>/submissions/tv --split test --subset gwhd2021-fsod-atsv
```
