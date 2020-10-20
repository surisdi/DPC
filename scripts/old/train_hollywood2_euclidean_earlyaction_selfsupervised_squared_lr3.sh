#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=7 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 16 \
--img_dim 80 \
--epochs 600 \
--early_action \
--early_action_self \
--pred_step 1 \
--lr 1e-3 \
--prefix hollywood2_euclidean_earlyaction_selfsupervised_lr3 \
--fp16
