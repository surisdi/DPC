#!/usr/bin/env bash
# This is the number 03 in the list
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9997 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--distance squared \
--network_feature resnet18 \
--dataset finegym \
--seq_len 5 \
--num_seq 6 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--lr 0.0001 \
--prefix finetune_earlyaction_self_eucl_poincare_finegym_64d \
--cross_gpu_score \
--early_action \
--early_action_self \
--feature_dim 64 \
--pretrain logs/log_train_dpc_kinetics_hyperv1_64d_euclidean/20201109_091723/model/model_best_epoch30.pth.tar \
--path_dataset /proj/vondrick/datasets/FineGym