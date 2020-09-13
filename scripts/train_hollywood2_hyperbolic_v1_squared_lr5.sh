#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 32 \
--img_dim 80 \
--epochs 600 \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'squared' \
--lr 1e-5 \
--prefix hyperbolic_hollywood_v1_squared_lr5 \
--resume /proj/vondrick/didac/code/DPC/dpc/log_hyperbolic_hyperbolic_v1_squared_lr5/hollywood2-80_r18_dpc-rnn_bs64_lr1e-05_seq8_pred3_len5_ds3_train-all/model/epoch300.pth.tar \
--fp16
