#!/bin/bash
set -e

# ==================== 配置 ====================

SPLIT="test"
GPUS=(0 1 2 3 4)
NUM_GPUS=${#GPUS[@]}

# 按 pair 数量均衡分配的子集（greedy bin-packing）
# GPU 0: 2290 pairs (1 subsets)
# GPU 1: 1958 pairs (4 subsets)
# GPU 2: 1948 pairs (4 subsets)
# GPU 3: 1960 pairs (5 subsets)
# GPU 4: 1975 pairs (6 subsets)
GPU0_SUBSETS=(
    x-ray-id-zfisb-fsod-dyjv
)
GPU1_SUBSETS=(
    paper-parts-fsod-rmrg
    dentalai-i4clz-fsod-fsuo
    trail-camera-fsod-egos
    wildfire-smoke-fsod-myxt
)
GPU2_SUBSETS=(
    recode-waste-czvmg-fsod-yxsw
    all-elements-fsod-mebv
    new-defects-in-wood-uewd1-fsod-tffp
    orionproducts-vtl2z-fsod-puhv
)
GPU3_SUBSETS=(
    flir-camera-objects-fsod-tdqp
    gwhd2021-fsod-atsv
    water-meter-jbktv-7vz5k-fsod-ftoz
    wb-prova-stqnm-fsod-rbvg
    the-dreidel-project-anzyr-fsod-zejm
)
GPU4_SUBSETS=(
    actions-zzid2-zb1hq-fsod-amih
    soda-bottles-fsod-haga
    defect-detection-yjplx-fxobh-fsod-amdi
    lacrosse-object-detection-fsod-uxkt
    aquarium-combined-fsod-gjvb
    aerial-airport-7ap9o-fsod-ddgc
)

PRED_TYPES=("tv" "text" "visual" "vqa")

# VQA rescoring 配置
VQA_RESCORE="--vqa-rescore"
OUTPUT_BASE="results/vqa-test"
SUBMIT_BASE="submission/vqa-test"

SESSION_PREFIX="fsod-test"

# ==================== 日志目录 ====================

LOG_DIR="scripts/parallel/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " Parallel Run (balanced) - $(date)"
echo " Split:   $SPLIT"
echo " GPUs:    ${GPUS[*]}"
echo " Logs:    $LOG_DIR"
echo "============================================"

# ==================== 杀掉旧会话（如有） ====================

for gi in "${!GPUS[@]}"; do
    SESSION="${SESSION_PREFIX}-gpu${GPUS[$gi]}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
done

# ==================== 启动 tmux 会话 ====================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# 函数：为指定 GPU 生成 worker 脚本并启动 tmux
launch_gpu() {
    local GPU_ID=$1
    shift
    local SUBSET_LIST=("$@")
    local NUM_ON_GPU=${#SUBSET_LIST[@]}

    local SESSION="${SESSION_PREFIX}-gpu${GPU_ID}"
    local LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"
    local WORKER_FILE="$LOG_DIR/worker_gpu${GPU_ID}.sh"

    cat > "$WORKER_FILE" <<EOF
#!/bin/bash
cd "$PROJECT_DIR"
conda activate sam3

echo "[GPU $GPU_ID] $NUM_ON_GPU subsets"
echo ""

EOF

    local COUNT=0
    for SUBSET in "${SUBSET_LIST[@]}"; do
        COUNT=$((COUNT + 1))
        cat >> "$WORKER_FILE" <<EOF
echo ">>>>>>>>>> [GPU $GPU_ID] ($COUNT/$NUM_ON_GPU) $SUBSET <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \\
    --subset "$SUBSET" \\
    --split "$SPLIT" \\
    --device cuda \\
    --output-dir "${OUTPUT_BASE}/${SUBSET}_${SPLIT}" \\
    $VQA_RESCORE
echo ">>>>>>>>>> [GPU $GPU_ID] $SUBSET DONE <<<<<<<<<<"

EOF
    done

    cat >> "$WORKER_FILE" <<EOF
echo ""
echo "[GPU $GPU_ID] All done."
EOF
    chmod +x "$WORKER_FILE"

    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
        "bash $LOG_DIR/worker_gpu${GPU_ID}.sh 2>&1 | tee $LOG_FILE"

    echo "  Launched session '$SESSION' -> $LOG_FILE"
}

launch_gpu 0 "${GPU0_SUBSETS[@]}"
launch_gpu 1 "${GPU1_SUBSETS[@]}"
launch_gpu 2 "${GPU2_SUBSETS[@]}"
launch_gpu 3 "${GPU3_SUBSETS[@]}"
launch_gpu 4 "${GPU4_SUBSETS[@]}"

echo ""
echo "All $NUM_GPUS tmux sessions running."
echo ""
echo "  查看进度:  tmux attach -t ${SESSION_PREFIX}-gpu<N>"
echo "  退出查看:  Ctrl+B, D"
echo "  列出会话:  tmux ls"
echo ""

# ==================== 等待完成 ====================

echo "Waiting for all sessions to finish..."

for gi in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$gi]}"
    SESSION="${SESSION_PREFIX}-gpu${GPU_ID}"

    while tmux has-session -t "$SESSION" 2>/dev/null; do
        sleep 5
    done

    if grep -qiE "error|traceback" "$LOG_DIR/gpu${GPU_ID}.log" 2>/dev/null; then
        echo "[WARN] GPU $GPU_ID log contains errors. Check $LOG_DIR/gpu${GPU_ID}.log"
    fi

    echo "[DONE] GPU $GPU_ID finished."
done

echo ""

# ==================== 收集提交文件 ====================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FOUND=0

ALL_SUBSETS=(
    "${GPU0_SUBSETS[@]}"
    "${GPU1_SUBSETS[@]}"
    "${GPU2_SUBSETS[@]}"
    "${GPU3_SUBSETS[@]}"
    "${GPU4_SUBSETS[@]}"
)

for PRED_TYPE in "${PRED_TYPES[@]}"; do
    SUBMIT_DIR="${SUBMIT_BASE}/${TIMESTAMP}_${PRED_TYPE}"
    mkdir -p "$SUBMIT_DIR"

    echo "  Collecting [$PRED_TYPE] ..."
    for SUBSET in "${ALL_SUBSETS[@]}"; do
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
echo " All subsets finished."
echo "============================================"
