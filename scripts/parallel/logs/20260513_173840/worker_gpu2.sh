#!/bin/bash
cd "/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge"
conda activate sam3

echo "[GPU 2] Subsets 9-12 (4 tasks)"
echo ""

echo ">>>>>>>>>> [GPU 2] (1/4) lacrosse-object-detection-fsod-uxkt <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --subset "lacrosse-object-detection-fsod-uxkt" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 2] lacrosse-object-detection-fsod-uxkt DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 2] (2/4) new-defects-in-wood-uewd1-fsod-tffp <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --subset "new-defects-in-wood-uewd1-fsod-tffp" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 2] new-defects-in-wood-uewd1-fsod-tffp DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 2] (3/4) orionproducts-vtl2z-fsod-puhv <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --subset "orionproducts-vtl2z-fsod-puhv" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 2] orionproducts-vtl2z-fsod-puhv DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 2] (4/4) paper-parts-fsod-rmrg <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --subset "paper-parts-fsod-rmrg" \
    --split "valid" \
    --device cuda
echo ">>>>>>>>>> [GPU 2] paper-parts-fsod-rmrg DONE <<<<<<<<<<"

echo ""
echo "[GPU 2] All done."
