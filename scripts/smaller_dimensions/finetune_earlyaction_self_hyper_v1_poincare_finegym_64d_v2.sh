#!/usr/bin/env bash
# This is the number 03 in the list
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9978 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--hyperbolic \
--hyperbolic_version 1 \
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
--fp64_hyper \
--num_workers 15 \
--lr 0.0001 \
--prefix finetune_earlyaction_self_hyper_v1_poincare_finegym_64d_v2 \
--cross_gpu_score \
--early_action \
--early_action_self \
--feature_dim 64 \
--pretrain logs/log_train_dpc_kinetics_hyperv1_64d/20201107_173841/model/model_best_epoch5.pth.tar \
--path_dataset /proj/vondrick/datasets/FineGym \
--seed 1