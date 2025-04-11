#!/bin/sh
# salloc -p bme_a10080g -c 10 --mem=50G --gres=gpu:1
# salloc -p bme_a10080g -N 1 -n 10 -t 5-00:00:00 --gres=gpu:1 --mem=50G

# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --if_biattn \
#     --resume --checkpoint_path logs/train_use_biattn/model_epoch_29.pth \
#     --writer logs/train_use_biattn

# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --num_layers 2 \
#     --if_biattn \
#     --writer logs/train_use_biattn_num_layers2

# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --num_layers 4 \
#     --if_biattn \
#     --writer logs/train_use_biattn_num_layers4

# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --num_layers 8 \
#     --if_biattn \
#     --writer logs/train_use_biattn_num_layers8

python decouple_bcr.py \
    --lr 0.0001 \
    --batch_size 16 \
    --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
    --embedding_dim 512 \
    --num_layers 2 \
    --if_biattn \
    --cell_type TransformerDecoder \
    --writer logs/train_use_biattn_num_layers2_transformerdecoder

# 1st no_biattn
# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --writer logs/train_no_biattn

# 2nd no_biattn num_layers=2
# python decouple_bcr.py \
#     --lr 0.0001 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --num_layers 2 \
#     --writer logs/train_no_biattn_num_layers2

# resume
# python decouple_bcr.py \
#     --lr 0.0001 \
#     --save_interval 10 \
#     --batch_size 128 \
#     --gpu 0 --sparsity_percentage 0.175 --epochs 150 \
#     --embedding_dim 512 \
#     --resume --checkpoint_path logs/train_no_biattn/model_epoch_3.pth \
#     --writer train_no_biattn2
