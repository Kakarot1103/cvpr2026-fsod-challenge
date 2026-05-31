#!/bin/bash
# 可视化预测结果 vs GT
# 用法:
#   bash scripts/vis.sh                                    # 全部子集
#   bash scripts/vis.sh --subsets gwhd2021-fsod-atsv       # 指定子集
#   bash scripts/vis.sh --score-threshold 0.3              # 过滤低分预测

python visualize_predictions.py \
    --submission-dir submission/test/vqa-test-ddp-new-prompt/20260518_232312_vqa \
    --output-dir results/visualizations \
    --split test \
    "$@"
