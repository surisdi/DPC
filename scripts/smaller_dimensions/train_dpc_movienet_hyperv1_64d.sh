#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9995 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--hyperbolic \
--hyperbolic_version 1 \
--distance squared \
--network_feature resnet18 \
--dataset movienet \
--seq_len 3 \
--num_seq 8 \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--fp16 \
--fp64_hyper \
--num_workers 15 \
--lr 0.001 \
--prefix train_dpc_movienet_hyperv1_64d \
--feature_dim 64 \
--cross_gpu_score \
--path_dataset /local/vondrick/didacsuris/local_data/MovieNet \
--seed 0 \
--pretrain logs/log_train_earlyaction_hyper_v1_poincare_movienet/20201031_183159/model/epoch41.pth.tar
