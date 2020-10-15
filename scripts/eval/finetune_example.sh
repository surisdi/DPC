#!/usr/bin/env bash
# In this example we implement the first option of evaluation, for the finetuning
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=7 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 80 \
--epochs 100 \
--hyperbolic \
--hyperbolic_version 1 \
--early_action \
--pred_step 0 \
--distance 'squared' \
--lr 1e-3 \
--prefix finetune_example \
--fp16 \
--fp64_hyper \
--finetune \
--finetune_path logs/log_hyperbolic_hollywood2_v1_earlyaction_selfsupervised_squared_lr3/20200920_102921/model/epoch600.pth.tar \
--finetune_input predictions \
--action_level_gt \
--n_classes 288

#--finetune_all False \
