#!/bin/sh
# salloc -p bme_a10080g -c 10 --mem=50G --gres=gpu:1

# python decouple_bcr.py \
#     --correlated 1 --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
#     --embedding_dim 512 \
#     --if_biattn \
#     --resume --checkpoint_path logs/train_use_biattn/model_epoch_29.pth \
#     --writer logs/train_use_biattn

# 1st no_biattn
# python decouple_bcr.py \
#     --correlated 1 --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
#     --embedding_dim 512 \
#     --writer logs/train_no_biattn

# 2nd no_biattn num_layers=2
python decouple_bcr.py \
    --correlated 1 --lr 0.0001 \
    --batch_size 128 \
    --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
    --embedding_dim 512 \
    --num_layers 2 \
    --writer logs/train_no_biattn_num_layers2

# resume
# python decouple_bcr.py \
#     --correlated 1 --lr 0.0001 \
#     --save_interval 10 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0 \
#     --embedding_dim 512 \
#     --resume --checkpoint_path logs/train_no_biattn/model_epoch_3.pth \
#     --writer train_no_biattn2
