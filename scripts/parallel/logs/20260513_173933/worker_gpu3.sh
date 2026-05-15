#!/bin/bash
cd "/home/chenzhigang/Projects/in-context-seg/cvpr2026-fsod-challenge"
conda activate sam3

echo "[GPU 3] Subsets 13-16 (4 tasks)"
echo ""

echo ">>>>>>>>>> [GPU 3] (1/4) recode-waste-czvmg-fsod-yxsw <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=3 python inference.py \
    --subset "recode-waste-czvmg-fsod-yxsw" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 3] recode-waste-czvmg-fsod-yxsw DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 3] (2/4) soda-bottles-fsod-haga <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=3 python inference.py \
    --subset "soda-bottles-fsod-haga" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 3] soda-bottles-fsod-haga DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 3] (3/4) the-dreidel-project-anzyr-fsod-zejm <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=3 python inference.py \
    --subset "the-dreidel-project-anzyr-fsod-zejm" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 3] the-dreidel-project-anzyr-fsod-zejm DONE <<<<<<<<<<"

echo ">>>>>>>>>> [GPU 3] (4/4) trail-camera-fsod-egos <<<<<<<<<<"
CUDA_VISIBLE_DEVICES=3 python inference.py \
    --subset "trail-camera-fsod-egos" \
    --split "test" \
    --device cuda
echo ">>>>>>>>>> [GPU 3] trail-camera-fsod-egos DONE <<<<<<<<<<"

echo ""
echo "[GPU 3] All done."
