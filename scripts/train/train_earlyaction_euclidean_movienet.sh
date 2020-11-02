#!/usr/bin/env bash
# This is the number 03 in the list
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--network_feature resnet18 \
--dataset movienet \
--seq_len 3 \
--num_seq 8 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--lr 0.001 \
--prefix train_earlyaction_euclidean_movienet \
--cross_gpu_score \
--early_action \
--early_action_self \
--path_dataset /local/vondrick/didacsuris/local_data/MovieNet