#!/bin/bash
cd "/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge"
conda activate sam3

echo "[GPU 0] Subsets 1-4 (4 tasks)"
echo ""

echo ">>>>>>>>>> [GPU 0] (1/4) actions-zzid2-zb1hq-fsod-amih <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --subset "actions-zzid2-zb1hq-fsod-amih" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 0] actions-zzid2-zb1hq-fsod-amih DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 0] (2/4) aerial-airport-7ap9o-fsod-ddgc <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --subset "aerial-airport-7ap9o-fsod-ddgc" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 0] aerial-airport-7ap9o-fsod-ddgc DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 0] (3/4) all-elements-fsod-mebv <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --subset "all-elements-fsod-mebv" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 0] all-elements-fsod-mebv DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 0] (4/4) aquarium-combined-fsod-gjvb <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --subset "aquarium-combined-fsod-gjvb" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 0] aquarium-combined-fsod-gjvb DONE <<<<<<<<<<"

echo ""
echo "[GPU 0] All done."
