#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 python main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 32 \
--img_dim 80 \
--epochs 600 \
--hyperbolic \
--hyp_cone \
--lr 1e-5 \
--prefix hyperbolic_hollywood_cone_lr3 \
--fp16