# CVPR 2026 FSOD Challenge

CVPR 2026 Few-Shot Object Detection (FSOD) Challenge solution based on SAM3.

## Overview

This project tackles few-shot object detection on the **RF-20VL** dataset using the **SAM3** foundation model. The core idea is to concatenate each support image with the query image, extract candidate bounding boxes from the query side via SAM3, and then refine detections through text-visual prompts.

The pipeline produces three types of predictions per image:
- **visual**: bounding boxes from visual prompts only
- **tv**: bounding boxes from text + visual prompts
- **text**: bounding boxes from text prompts only

## Project Structure

```
.
├── inference.py          # Main inference & evaluation pipeline
├── evaluate.py           # Standalone evaluation script for submissions
├── model/
│   └── sam3.py           # Sam3Segmenter wrapper class
├── datasets/
│   └── rf20vl.py         # RF-20VL dataset loader
├── sam3/                 # SAM3 library (source)
├── scripts/
│   ├── run_single.sh     # Run inference on a single subset
│   ├── run_all.sh        # Run inference on all subsets
│   ├── run_loop.sh       # Loop over all subsets sequentially
│   └── parallel/         # Multi-GPU parallel inference scripts
├── data/                 # RF-20VL dataset (symlink, excluded from git)
├── results/              # Inference results (excluded from git)
└── submission/           # Submission pickle files (excluded from git)
```

## Setup

### Requirements

- Python 3.10+
- PyTorch with CUDA
- pycocotools
- torchvision
- PIL (Pillow)

### Data

Place the RF-20VL dataset at `./data` (or create a symlink):

```bash
ln -s /path/to/RF-20VL ./data
```

### Pretrained Model

The SAM3 pretrained model should be available at:
`/data/chenzhigang/Pretrained_models/SAM/sam3/sam3.pt`

## Usage

### Run on a single subset

```bash
bash scripts/run_single.sh
```

Edit `SUBSET` and `SPLIT` variables in the script as needed.

### Run on all subsets

```bash
bash scripts/run_all.sh
```

### Multi-GPU parallel inference

```bash
bash scripts/parallel/run_parallel.sh
```

### Evaluate a submission

```bash
python evaluate.py --submission results/<timestamp>/submissions/tv
python evaluate.py --submission results/<timestamp>/submissions/tv --split test
python evaluate.py --submission results/<timestamp>/submissions/tv --subset gwhd2021-fsod-atsv
```

### Custom inference

```bash
python inference.py \
    --data-path ./data \
    --split test \
    --subset <subset-name> \
    --device cuda \
    --nms-iou 0.8
```

## Method

1. For each (query_image, category) pair, load all support images from the train split
2. Concatenate each support image with the query image horizontally
3. Use SAM3 to predict candidate boxes on the concatenated image, filtering for the query half
4. Merge and deduplicate candidate boxes across all supports via NMS
5. Run SAM3 on the query image with candidate boxes as geometric prompts + category name as text prompt
6. Evaluate with COCO metrics (AP, AP@50, AP@75, etc.)
