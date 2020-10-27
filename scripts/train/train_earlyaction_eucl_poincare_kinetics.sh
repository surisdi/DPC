#!/usr/bin/env bash
# This is the number 03 in the list
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 8 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--lr 0.001 \
--prefix train_earlyaction_eucl_poincare_kinetics \
--cross_gpu_score \
--early_action \
--early_action_self