#!/usr/bin/env bash
# This is the number 01 in the list
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 8 \
--ds 3 \
--batch_size 8 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--cross_gpu_score \
--lr 0.001 \
--prefix train_dpc_eucl_poincare_kinetics
