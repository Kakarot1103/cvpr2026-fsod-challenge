#!/bin/bash
set -e

# ==================== 配置 ====================

SPLIT="test"
# SPLIT="valid"

GPUS=(0 1 2 3 4)

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

PRED_TYPES=("tv" "text" "visual" "vqa")

# VQA rescoring 配置
VQA_RESCORE="--vqa-rescore"
VQA_TARGET="--vqa-target tv"
OUTPUT_BASE="results/test/text_on_cat"
SUBMIT_BASE="submission/test/text_on_cat"

NUM_GPUS=${#GPUS[@]}
TOTAL=${#SUBSETS[@]}
SESSION="fsod-ddp"

# ==================== 日志目录 ====================

LOG_DIR="scripts/parallel/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " DDP Run - $(date)"
echo " Split:   $SPLIT"
echo " GPUs:    ${GPUS[*]} ($NUM_GPUS GPUs)"
echo " Subsets: $TOTAL"
echo " Logs:    $LOG_DIR"
echo "============================================"

# ==================== 杀掉旧会话（如有） ====================

tmux kill-session -t "$SESSION" 2>/dev/null || true

# ==================== 构造 CUDA_VISIBLE_DEVICES ====================

GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

# ==================== 生成 worker 脚本 ====================

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKER_FILE="$LOG_DIR/worker.sh"

cat > "$WORKER_FILE" <<EOF
#!/bin/bash
cd "$PROJECT_DIR"
# Initialize conda (auto-detect installation)
eval "$(conda shell.bash hook 2>/dev/null)" || source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate sam3

echo "DDP Inference: $TOTAL subsets, $NUM_GPUS GPUs"
echo "CUDA_VISIBLE_DEVICES=$GPU_LIST"
echo ""

EOF

COUNT=0
for SUBSET in "${SUBSETS[@]}"; do
    COUNT=$((COUNT + 1))
    cat >> "$WORKER_FILE" <<EOF
echo ">>>>>>>>>> ($COUNT/$TOTAL) $SUBSET <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \\
    --nproc_per_node=$NUM_GPUS \\
    --master_port 29500 \\
    inference_ddp.py \\
    --subset "$SUBSET" \\
    --split "$SPLIT" \\
    --device cuda \\
    --output-dir "${OUTPUT_BASE}/${SUBSET}_${SPLIT}" \\
    $VQA_RESCORE \\
    $VQA_TARGET
echo ">>>>>>>>>> $SUBSET DONE <<<<<<<<<<"

EOF
done

cat >> "$WORKER_FILE" <<EOF
echo ""
echo "All $TOTAL subsets finished."
EOF
chmod +x "$WORKER_FILE"

# ==================== 启动 tmux 会话 ====================

tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
    "bash $WORKER_FILE 2>&1 | tee $LOG_DIR/ddp.log"

echo "  Launched session '$SESSION' -> $LOG_DIR/ddp.log"
echo ""
echo "  查看进度:  tmux attach -t $SESSION"
echo "  退出查看:  Ctrl+B, D"
echo ""

# ==================== 等待完成 ====================

echo "Waiting for session to finish..."

while tmux has-session -t "$SESSION" 2>/dev/null; do
    sleep 5
done

if grep -qiE "error|traceback" "$LOG_DIR/ddp.log" 2>/dev/null; then
    echo "[WARN] Log contains errors. Check $LOG_DIR/ddp.log"
fi

echo "[DONE] Session finished."

# ==================== 收集提交文件 ====================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FOUND=0

for PRED_TYPE in "${PRED_TYPES[@]}"; do
    SUBMIT_DIR="${SUBMIT_BASE}/${TIMESTAMP}_${PRED_TYPE}"
    mkdir -p "$SUBMIT_DIR"

    echo "  Collecting [$PRED_TYPE] ..."
    for SUBSET in "${SUBSETS[@]}"; do
        CANDIDATE="${OUTPUT_BASE}/${SUBSET}_${SPLIT}/submissions/$PRED_TYPE/${SUBSET}.pkl"
        if [ -f "$CANDIDATE" ]; then
            cp "$CANDIDATE" "$SUBMIT_DIR/${SUBSET}.pkl"
            echo "    ${SUBSET}.pkl"
            FOUND=1
        else
            echo "    MISSING: ${SUBSET}.pkl"
        fi
    done
done

echo ""
if [ $FOUND -eq 1 ]; then
    echo "Submission files saved to: ${SUBMIT_BASE}/${TIMESTAMP}_*"
else
    echo "WARNING: No submission files found."
fi

echo ""
echo "============================================"
echo " All $TOTAL subsets finished."
echo "============================================"
