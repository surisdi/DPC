#!/usr/bin/env bash
# This is the number 03 in the list
CUDA_VISIBLE_DEVICES=4 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9999 \
--nproc_per_node=1 \
main.py \
--pred_step 3 \
--hyperbolic \
--hyperbolic_version 2 \
--distance squared \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 8 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 15 \
--lr 0.001 \
--prefix train_dpc_hyper_v2_poincare_kinetics \
--path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data
