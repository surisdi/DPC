#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 8 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--fp16 \
--num_workers 15 \
--lr 0.001 \
--prefix train_dpc_kinetics_hyperv1_64d_euclidean \
--feature_dim 64 \
--cross_gpu_score \
--path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data \
--seed 0 \
--pretrain logs/log_train_earlyaction_eucl_poincare_kinetics/20201025_122942/model/model_best_epoch9.pth.tar \
--partial 0.1 \
--resume logs/log_train_dpc_kinetics_hyperv1_64d_euclidean/20201109_091723/model/model_best_epoch7.pth.tar
