#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=4 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 10 \
--img_dim 80 \
--epochs 600 \
--hyperbolic \
--hyperbolic_version 1 \
--early_action \
--early_action_self \
--pred_step 1 \
--distance 'squared' \
--lr 1e-3 \
--prefix hyperbolic_hollywood2_v1_earlyaction_selfsupervised_squared_lr3 \
--fp16
