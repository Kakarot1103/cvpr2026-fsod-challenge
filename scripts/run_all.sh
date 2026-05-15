#!/bin/bash
set -e

# ==================== 配置 ====================

# 数据划分
SPLIT="test"
# SPLIT="valid"

# 设备
DEVICE="cuda"
# DEVICE="cpu"

# ==================== 运行 ====================

echo "============================================"
echo " All Subsets Inference"
echo " Split:  $SPLIT"
echo " Device: $DEVICE"
echo "============================================"

python inference.py \
    --split "$SPLIT" \
    --device "$DEVICE"
