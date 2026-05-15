#!/bin/bash
set -e

# ==================== 配置 ====================

SPLIT="test"
# SPLIT="valid"

PRED_TYPES=("tv" "text" "visual")

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

# ==================== 收集 ====================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOTAL=${#SUBSETS[@]}
FOUND=0
MISSING=0

echo "============================================"
echo " Collect Submissions"
echo " Split: $SPLIT"
echo " Total: $TOTAL subsets"
echo "============================================"
echo ""

for PRED_TYPE in "${PRED_TYPES[@]}"; do
    SUBMIT_DIR="submission/${TIMESTAMP}_${PRED_TYPE}"
    mkdir -p "$SUBMIT_DIR"

    echo "[$PRED_TYPE]"
    for SUBSET in "${SUBSETS[@]}"; do
        SRC=""
        for DIR in results/*_${SPLIT}; do
            CANDIDATE="$DIR/submissions/$PRED_TYPE/${SUBSET}.pkl"
            if [ -f "$CANDIDATE" ]; then
                SRC="$CANDIDATE"
                break
            fi
        done
        if [ -n "$SRC" ]; then
            cp "$SRC" "$SUBMIT_DIR/${SUBSET}.pkl"
            echo "  OK   ${SUBSET}.pkl"
            FOUND=$((FOUND + 1))
        else
            echo "  MISS ${SUBSET}.pkl"
            MISSING=$((MISSING + 1))
        fi
    done
    echo ""
done

echo "============================================"
echo " Found:    $FOUND"
echo " Missing:  $MISSING"
echo " Saved to: submission/${TIMESTAMP}_*"
echo "============================================"
echo ""

# ==================== 压缩 ====================

echo "Compressing..."
for PRED_TYPE in "${PRED_TYPES[@]}"; do
    SUBMIT_DIR="submission/${TIMESTAMP}_${PRED_TYPE}"
    if [ -d "$SUBMIT_DIR" ]; then
        ZIP_FILE="submission/${TIMESTAMP}_${PRED_TYPE}.zip"
        (cd "$SUBMIT_DIR" && zip -q "../${TIMESTAMP}_${PRED_TYPE}.zip" ./*.pkl)
        echo "  ${TIMESTAMP}_${PRED_TYPE}.zip"
    fi
done

echo ""
echo "Done. ZIP files in submission/"
