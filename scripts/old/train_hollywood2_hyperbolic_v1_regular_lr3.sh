#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 8 \
--img_dim 80 \
--epochs 300 \
--start-epoch 0 \
--reset_lr \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'regular' \
--lr 1e-3 \
--prefix hyperbolic_hollywood_v1_regular_lr3 \
--fp16
