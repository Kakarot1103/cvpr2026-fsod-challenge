#!/bin/bash
set -e

# ==================== 配置 ====================

# DetPO prompt 目录
PROMPT_DIR="DetPO/prompts/detpo/Qwen3-VL-8B-Instruct"

# 输出目录
OUTPUT_DIR="prompts/sam3_prompt_mapping"

# 每个类别的候选 prompt 数量
NUM_CANDIDATES=5

# LLM 服务配置
LLM_SERVER="http://localhost:22002/v1"
LLM_MODEL="qwen3-vl-8b"

# 设备
DEVICE="cuda"

# ==================== 运行 ====================

echo "============================================"
echo " Generate SAM3 Prompt Mapping (All Subsets)"
echo " Candidates:  $NUM_CANDIDATES"
echo " Output:      $OUTPUT_DIR"
echo " LLM Server:  $LLM_SERVER"
echo " Device:      $DEVICE"
echo "============================================"

python generate_prompt_mapping.py \
    --prompt-dir "$PROMPT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-candidate-prompts "$NUM_CANDIDATES" \
    --llm-server-url "$LLM_SERVER" \
    --llm-model "$LLM_MODEL" \
    --device "$DEVICE"
