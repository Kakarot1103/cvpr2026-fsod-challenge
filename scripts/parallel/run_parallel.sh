#!/bin/bash
set -e

# ==================== 配置 ====================

SPLIT="test"
# SPLIT="valid"

GPUS=(0 1 2 3 4)
NUM_GPUS=${#GPUS[@]}

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

TOTAL=${#SUBSETS[@]}
PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

PRED_TYPES=("tv" "text" "visual")

SESSION_PREFIX="fsod"

# ==================== 日志目录 ====================

LOG_DIR="scripts/parallel/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo " Parallel Run - $(date)"
echo " Split:   $SPLIT"
echo " GPUs:    ${GPUS[*]}"
echo " Subsets: $TOTAL (${PER_GPU} per GPU)"
echo " Logs:    $LOG_DIR"
echo "============================================"

# ==================== 杀掉旧会话（如有） ====================

for gi in "${!GPUS[@]}"; do
    SESSION="${SESSION_PREFIX}-gpu${GPUS[$gi]}"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
done

# ==================== 启动 tmux 会话 ====================

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

for gi in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$gi]}"
    START=$((gi * PER_GPU))
    END=$((START + PER_GPU))
    [ $END -gt $TOTAL ] && END=$TOTAL

    SESSION="${SESSION_PREFIX}-gpu${GPU_ID}"
    LOG_FILE="$LOG_DIR/gpu${GPU_ID}.log"

    # 构造该 GPU 要跑的子集列表
    SUBSET_ARGS=""
    for si in $(seq $START $((END - 1))); do
        SUBSET_ARGS="$SUBSET_ARGS ${SUBSETS[$si]}"
    done

    # 写入 worker 脚本（在 tmux 里执行）—— 直接展开每个子集命令，避免变量问题
    WORKER_FILE="$LOG_DIR/worker_gpu${GPU_ID}.sh"
    NUM_ON_GPU=$((END - START))

    cat > "$WORKER_FILE" <<EOF
#!/bin/bash
cd "$PROJECT_DIR"
conda activate sam3

echo "[GPU $GPU_ID] Subsets $((START+1))-$END ($NUM_ON_GPU tasks)"
echo ""

EOF

    COUNT=0
    for si in $(seq $START $((END - 1))); do
        COUNT=$((COUNT + 1))
        SUBSET="${SUBSETS[$si]}"
        cat >> "$WORKER_FILE" <<EOF
echo ">>>>>>>>>> [GPU $GPU_ID] ($COUNT/$NUM_ON_GPU) $SUBSET <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py \\
    --subset "$SUBSET" \\
    --split "$SPLIT" \\
    --device cuda
echo ">>>>>>>>>> [GPU $GPU_ID] $SUBSET DONE <<<<<<<<<<"

EOF
    done

    cat >> "$WORKER_FILE" <<EOF
echo ""
echo "[GPU $GPU_ID] All done."
EOF
    chmod +x "$WORKER_FILE"

    # 启动 tmux 会话，同时输出到终端和日志文件
    tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
        "bash $LOG_DIR/worker_gpu${GPU_ID}.sh 2>&1 | tee $LOG_FILE"

    echo "  Launched session '$SESSION' -> $LOG_FILE"
done

echo ""
echo "All $NUM_GPUS tmux sessions running."
echo ""
echo "  查看进度:  tmux attach -t ${SESSION_PREFIX}-gpu<N>"
echo "  退出查看:  Ctrl+B, D"
echo "  列出会话:  tmux ls"
echo ""

# ==================== 等待完成 ====================

echo "Waiting for all sessions to finish..."

FAIL=0
for gi in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$gi]}"
    SESSION="${SESSION_PREFIX}-gpu${GPU_ID}"

    # 等待该 tmux 会话结束
    while tmux has-session -t "$SESSION" 2>/dev/null; do
        sleep 5
    done

    # 检查日志中是否有错误标记（可选）
    if grep -qiE "error|traceback" "$LOG_DIR/gpu${GPU_ID}.log" 2>/dev/null; then
        echo "[WARN] GPU $GPU_ID log contains errors. Check $LOG_DIR/gpu${GPU_ID}.log"
    fi

    echo "[DONE] GPU $GPU_ID finished."
done

echo ""

# ==================== 收集提交文件 ====================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FOUND=0

for PRED_TYPE in "${PRED_TYPES[@]}"; do
    SUBMIT_DIR="submission/${TIMESTAMP}_${PRED_TYPE}"
    mkdir -p "$SUBMIT_DIR"

    echo "  Collecting [$PRED_TYPE] ..."
    for SUBSET in "${SUBSETS[@]}"; do
        # 每个 subset 的 results 目录不同，需遍历所有匹配目录查找
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
            echo "    ${SUBSET}.pkl"
            FOUND=1
        else
            echo "    MISSING: ${SUBSET}.pkl"
        fi
    done
done

echo ""
if [ $FOUND -eq 1 ]; then
    echo "Submission files saved to: submission/${TIMESTAMP}_*"
else
    echo "WARNING: No submission files found."
fi

echo ""
echo "============================================"
echo " All $TOTAL subsets finished."
echo "============================================"
