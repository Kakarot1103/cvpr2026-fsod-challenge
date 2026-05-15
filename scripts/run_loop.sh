#!/bin/bash
set -e

# ==================== 配置 ====================

# 数据划分
SPLIT="test"
# SPLIT="valid"

# 设备
DEVICE="cuda"
# DEVICE="cpu"

# 所有子集
SUBSETS=(
    actions-zzid2-zb1hq-fsod-amih
    aerial-airport-7ap9o-fsod-ddgc
    all-elements-fsod-mebv
    aquarium-combined-fsod-gjvb
    defect-detection-yjplx-fxobh-fsod-amdi
    dentalai-i4clz-fsod-fsuo
    flir-camera-objects-fsod-tdqp
    gwhd2021-fsod-atsv
    lacrosse-object-detection-fsod-uxkt
    new-defects-in-wood-uewd1-fsod-tffp
    orionproducts-vtl2z-fsod-puhv
    paper-parts-fsod-rmrg
    recode-waste-czvmg-fsod-yxsw
    soda-bottles-fsod-haga
    the-dreidel-project-anzyr-fsod-zejm
    trail-camera-fsod-egos
    water-meter-jbktv-7vz5k-fsod-ftoz
    wb-prova-stqnm-fsod-rbvg
    wildfire-smoke-fsod-myxt
    x-ray-id-zfisb-fsod-dyjv
)

# ==================== 提交类型 ====================
PRED_TYPES=("tv" "text" "visual")

# ==================== 运行 ====================

TOTAL=${#SUBSETS[@]}
RESULTS_DIR=""

echo "============================================"
echo " Loop All Subsets"
echo " Split:  $SPLIT"
echo " Device: $DEVICE"
echo " Total:  $TOTAL subsets"
echo "============================================"

for i in "${!SUBSETS[@]}"; do
    SUBSET="${SUBSETS[$i]}"
    echo ""
    echo ">>>>>>>>>> [$((i + 1))/$TOTAL] $SUBSET <<<<<<<<<<"

    python inference.py \
        --subset "$SUBSET" \
        --split "$SPLIT" \
        --device "$DEVICE"

    # 记录第一次运行生成的 results 目录
    if [ -z "$RESULTS_DIR" ]; then
        RESULTS_DIR=$(ls -dt results/*_${SPLIT} 2>/dev/null | head -1)
    fi

    echo ">>>>>>>>>> [$((i + 1))/$TOTAL] $SUBSET DONE <<<<<<<<<<"
done

echo ""
echo "============================================"
echo " All $TOTAL subsets finished."
echo "============================================"

# --- 收集提交文件 ---
if [ -n "$RESULTS_DIR" ] && [ -d "$RESULTS_DIR/submissions" ]; then
    TIMESTAMP=$(basename "$RESULTS_DIR" | cut -d_ -f1)

    for PRED_TYPE in "${PRED_TYPES[@]}"; do
        SUBMIT_DIR="submission/${TIMESTAMP}_${PRED_TYPE}"
        mkdir -p "$SUBMIT_DIR"

        echo "  Collecting [$PRED_TYPE] ..."
        for SUBSET in "${SUBSETS[@]}"; do
            SRC="$RESULTS_DIR/submissions/$PRED_TYPE/${SUBSET}.pkl"
            if [ -f "$SRC" ]; then
                cp "$SRC" "$SUBMIT_DIR/${SUBSET}.pkl"
                echo "    ${SUBSET}.pkl"
            else
                echo "    MISSING: ${SUBSET}.pkl"
            fi
        done
    done

    echo ""
    echo "Submission files saved to: submission/${TIMESTAMP}_*"
else
    echo "WARNING: No submission files found."
fi
