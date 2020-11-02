#!/usr/bin/env bash
# The kinetics was trained with poincare
CUDA_VISIBLE_DEVICES=0 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9991 \
--nproc_per_node=1 \
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
--batch_size 16 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 15 \
--lr 0.0001 \
--prefix finetune_earlyaction_self_hyper_v1_cone_finegym_frompoincare_margin3 \
--cross_gpu_score \
--early_action \
--early_action_self \
--hyp_cone \
--margin 3 \
--pretrain logs/log_train_earlyaction_hyper_v1_poincare_kinetics_lr4/20201023_151021/model/model_best_epoch31.pth.tar
#--pretrain logs/log_train_dpc_hyper_v1_poincare_kinetics/20201019_195227/model/model_best_epoch159.pth.tar
