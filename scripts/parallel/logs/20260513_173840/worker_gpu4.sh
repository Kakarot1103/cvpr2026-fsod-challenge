#!/bin/bash
cd "/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge"
conda activate sam3

echo "[GPU 4] Subsets 17-20 (4 tasks)"
echo ""

echo ">>>>>>>>>> [GPU 4] (1/4) water-meter-jbktv-7vz5k-fsod-ftoz <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=4 python inference.py \
    --subset "water-meter-jbktv-7vz5k-fsod-ftoz" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 4] water-meter-jbktv-7vz5k-fsod-ftoz DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 4] (2/4) wb-prova-stqnm-fsod-rbvg <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=4 python inference.py \
    --subset "wb-prova-stqnm-fsod-rbvg" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 4] wb-prova-stqnm-fsod-rbvg DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 4] (3/4) wildfire-smoke-fsod-myxt <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=4 python inference.py \
    --subset "wildfire-smoke-fsod-myxt" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 4] wildfire-smoke-fsod-myxt DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 4] (4/4) x-ray-id-zfisb-fsod-dyjv <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=4 python inference.py \
    --subset "x-ray-id-zfisb-fsod-dyjv" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 4] x-ray-id-zfisb-fsod-dyjv DONE <<<<<<<<<<"

echo ""
echo "[GPU 4] All done."
