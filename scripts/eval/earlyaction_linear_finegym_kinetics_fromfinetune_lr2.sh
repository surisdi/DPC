#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9985 \
--nproc_per_node=4 \
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
--lr 1e-2 \
--prefix earlyaction_linear_finegym_kinetics_fromfinetune_lr2 \
--fp16 \
--fp64_hyper \
--pretrain logs/log_finetune_earlyaction_self_hyper_v1_poincare_finegym/20201025_164701/model/model_best_epoch114.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--use_labels \
--num_workers 15 \
--only_train_linear \
--early_action \
--num_workers 8 \
--seed 0 \
--action_level_gt \
--path_dataset /proj/vondrick/datasets/FineGym

