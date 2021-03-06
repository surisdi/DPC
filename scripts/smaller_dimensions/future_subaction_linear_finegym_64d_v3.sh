#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9983 \
--nproc_per_node=3 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--hyperbolic \
--hyperbolic_version 1 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--distance 'squared' \
--lr 1e-3 \
--prefix future_subaction_linear_finegym_64d_v3 \
--fp16 \
--fp64_hyper \
--pretrain logs/log_finetune_earlyaction_self_hyper_v1_poincare_finegym_64d/20201109_201936/model/model_best_epoch199.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 15 \
--pred_future \
--num_workers 8 \
--seed 0 \
--feature_dim 64 \
--path_dataset /proj/vondrick/datasets/FineGym

