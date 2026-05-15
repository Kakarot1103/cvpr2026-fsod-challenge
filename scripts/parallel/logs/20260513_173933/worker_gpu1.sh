#!/bin/bash
cd "/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge"
conda activate sam3

echo "[GPU 1] Subsets 5-8 (4 tasks)"
echo ""

echo ">>>>>>>>>> [GPU 1] (1/4) defect-detection-yjplx-fxobh-fsod-amdi <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --subset "defect-detection-yjplx-fxobh-fsod-amdi" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 1] defect-detection-yjplx-fxobh-fsod-amdi DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 1] (2/4) dentalai-i4clz-fsod-fsuo <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --subset "dentalai-i4clz-fsod-fsuo" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 1] dentalai-i4clz-fsod-fsuo DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 1] (3/4) flir-camera-objects-fsod-tdqp <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --subset "flir-camera-objects-fsod-tdqp" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 1] flir-camera-objects-fsod-tdqp DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 1] (4/4) gwhd2021-fsod-atsv <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --subset "gwhd2021-fsod-atsv" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 1] gwhd2021-fsod-atsv DONE <<<<<<<<<<"

echo ""
echo "[GPU 1] All done."
