#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=4 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 8 \
--img_dim 80 \
--epochs 600 \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'squared' \
--lr 1e-5 \
--prefix hyperbolic_hollywood_v1_squared_lr5 \
--fp16
