#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9997 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 32 \
--img_dim 80 \
--epochs 600 \
--hyperbolic \
--hyperbolic_version 1 \
--early_action \
--pred_step 0 \
--lr 1e-3 \
--prefix hyperbolic_hollywood2_v1_earlyaction_supervised_lr3 \
--n_classes 12 \
--fp16
