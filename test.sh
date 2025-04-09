#!/bin/sh

python decouple_bcr.py \
    --correlated 1 --lr 0.0001 \
    --save_interval 1 \
    --batch_size 128 \
    --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
    --embedding_dim 512 \
    --test \
    --checkpoint_path 'train_no_biattn/model_epoch_0.pth'