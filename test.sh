#!/bin/sh

# python decouple_bcr.py \
#     --correlated 1 --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
#     --embedding_dim 512 \
#     --if_biattn \
#     --test \
#     --checkpoint_path 'logs/train_use_biattn/model_epoch_19.pth'


python decouple_bcr.py \
    --correlated 1 --lr 0.0001 \
    --batch_size 128 \
    --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
    --embedding_dim 512 \
    --test \
    --checkpoint_path logs/train_no_biattn/model_epoch_109.pth