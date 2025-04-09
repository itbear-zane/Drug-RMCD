#!/bin/sh
# salloc -p bme_a10080g -c 10 --mem=50G --gres=gpu:1
# python decouple_bcr.py \
#     --correlated 1 --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
#     --embedding_dim 512 \
#     --if_biattn \
#     --writer train_use_biattn

python decouple_bcr.py \
    --correlated 1 --lr 0.0001 \
    --batch_size 16 \
    --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
    --embedding_dim 512 \
    --writer train_no_biattn
