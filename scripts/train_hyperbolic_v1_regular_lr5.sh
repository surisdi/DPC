#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9994 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset k600 \
--batch_size 64 \
--img_dim 80 \
--epochs 300 \
--start-epoch 0 \
--reset_lr \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'regular' \
--lr 1e-5 \
--prefix hyperbolic_v1_regular_lr5
