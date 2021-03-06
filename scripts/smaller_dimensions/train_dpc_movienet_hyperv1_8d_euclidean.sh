#!/usr/bin/env bash
# Reduced to 0.0001 because it was stuck
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--network_feature resnet18 \
--dataset movienet \
--seq_len 3 \
--num_seq 8 \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--fp16 \
--num_workers 15 \
--lr 0.001 \
--prefix train_dpc_movienet_hyperv1_8d_euclidean \
--feature_dim 8 \
--cross_gpu_score \
--path_dataset /local/vondrick/didacsuris/local_data/MovieNet \
--seed 0 \
--pretrain logs/log_train_earlyaction_euclidean_movienet/20201102_141930/model/epoch49.pth.tar \
--resume logs/log_train_dpc_movienet_hyperv1_8d_euclidean/20201108_144943/model/epoch2.pth.tar
