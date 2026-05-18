#!/bin/bash
# 跑完 RF-20VL 全部 20 个子集的 DetPO 评估
# 用法: 修改下方 GPU 配置后 bash scripts/run_all_subsets.sh [OUTPUT_DIR]

set -euo pipefail

# ============ 配置 ============
export CUDA_VISIBLE_DEVICES=5
MODEL_NAME="Qwen3-VL-8B-Instruct"
SERVED_MODEL_NAME="qwen3-vl-8b"
ROOT_PATH="/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge/data"
DATA_INSTR_PATH="prompts/detpo/Qwen3-VL-8B-Instruct/all_refined_class_instructions"
SERVER_URL="http://localhost:22002/v1"
OUTPUT_DIR="${1:-results/run_all_$(date +%Y%m%d_%H%M%S)}"
# ==============================

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

# 收集所有子集
mapfile -t SUBSETS < <(ls "$ROOT_PATH" | grep -v README)

echo "============================================"
echo " DetPO 全子集评估"
echo " 子集数量: ${#SUBSETS[@]}"
echo " 输出目录: $OUTPUT_DIR"
echo "============================================"

# 检查 vLLM 服务是否可用
if ! curl -s "$SERVER_URL/models" > /dev/null 2>&1; then
    echo "错误: vLLM 服务 ($SERVER_URL) 不可达，请先启动服务"
    exit 1
fi

TOTAL=${#SUBSETS[@]}
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!SUBSETS[@]}"; do
    SUBSET="${SUBSETS[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "========== [$IDX/$TOTAL] $SUBSET =========="

    SUB_START=$(date +%s)

    if PYTHONPATH=. python detpo/run_evaluation.py \
        --model_name "$MODEL_NAME" \
        --served_model_name "$SERVED_MODEL_NAME" \
        --root_path "$ROOT_PATH/" \
        --dataset_path "$SUBSET" \
        --data_instr_path "$DATA_INSTR_PATH" \
        --server_url "$SERVER_URL" \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/logs_${SUBSET}.txt" | tail -5; then

        SUB_END=$(date +%s)
        SUB_ELAPSED=$((SUB_END - SUB_START))
        SUCCESS=$((SUCCESS + 1))
        echo "[$IDX/$TOTAL] $SUBSET 完成 (${SUB_ELAPSED}s)"
    else
        SUB_END=$(date +%s)
        SUB_ELAPSED=$((SUB_END - SUB_START))
        FAILED=$((FAILED + 1))
        echo "[$IDX/$TOTAL] $SUBSET 失败 (${SUB_ELAPSED}s)"
    fi
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(( (TOTAL_ELAPSED % 3600) / 60 ))

echo ""
echo "============================================"
echo " 全部完成"
echo " 成功: $SUCCESS / $TOTAL"
echo " 失败: $FAILED / $TOTAL"
echo " 总耗时: ${HOURS}h ${MINS}m"
echo " 结果目录: $OUTPUT_DIR"
echo "============================================"

# 汇总各子集 mAP
echo ""
echo "各子集 mAP@50-95 汇总:"
echo "-------------------------------"
for SUBSET in "${SUBSETS[@]}"; do
    EVAL_FILE="$OUTPUT_DIR/evaluations/default/evaluation_${SUBSET}.json"
    if [ -f "$EVAL_FILE" ]; then
        MAP=$(python3 -c "import json; print(f'{json.load(open(\"$EVAL_FILE\"))[\"model\"][0]:.4f}')")
        echo "  $SUBSET: $MAP"
    else
        echo "  $SUBSET: 未生成结果"
    fi
done

# 计算平均 mAP
echo "-------------------------------"
python3 -c "
import json, os, glob
pattern = '$OUTPUT_DIR/evaluations/default/evaluation_*.json'
files = glob.glob(pattern)
if files:
    maps = [json.load(open(f))['model'][0] for f in files]
    print(f'平均 mAP@50-95: {sum(maps)/len(maps):.4f} ({len(maps)} 个子集)')
else:
    print('无评估结果文件')
"
