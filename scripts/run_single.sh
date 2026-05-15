#!/bin/bash
set -e

# ==================== 配置 ====================

# 子集（取消注释你要跑的那个）
# SUBSET="actions-zzid2-zb1hq-fsod-amih"
SUBSET="aerial-airport-7ap9o-fsod-ddgc"
# SUBSET="all-elements-fsod-mebv"
# SUBSET="aquarium-combined-fsod-gjvb"
# SUBSET="defect-detection-yjplx-fxobh-fsod-amdi"
# SUBSET="dentalai-i4clz-fsod-fsuo"
# SUBSET="flir-camera-objects-fsod-tdqp"
# SUBSET="gwhd2021-fsod-atsv"
# SUBSET="lacrosse-object-detection-fsod-uxkt"
# SUBSET="new-defects-in-wood-uewd1-fsod-tffp"
# SUBSET="orionproducts-vtl2z-fsod-puhv"
# SUBSET="paper-parts-fsod-rmrg"
# SUBSET="recode-waste-czvmg-fsod-yxsw"
# SUBSET="soda-bottles-fsod-haga"
# SUBSET="the-dreidel-project-anzyr-fsod-zejm"
# SUBSET="trail-camera-fsod-egos"
# SUBSET="water-meter-jbktv-7vz5k-fsod-ftoz"
# SUBSET="wb-prova-stqnm-fsod-rbvg"
# SUBSET="wildfire-smoke-fsod-myxt"
# SUBSET="x-ray-id-zfisb-fsod-dyjv"

# 数据划分
# SPLIT="test"
SPLIT="valid"

# 设备
DEVICE="cuda"
# DEVICE="cpu"

# ==================== 运行 ====================

echo "============================================"
echo " Single Subset Inference"
echo " Subset:  $SUBSET"
echo " Split:   $SPLIT"
echo " Device:  $DEVICE"
echo "============================================"

python inference.py \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --device "$DEVICE"
