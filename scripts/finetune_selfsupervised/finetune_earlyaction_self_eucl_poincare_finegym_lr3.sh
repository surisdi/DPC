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
--num_workers 8 \
--lr 0.001 \
--prefix finetune_earlyaction_self_eucl_poincare_finegym_lr3 \
--cross_gpu_score \
--early_action \
--early_action_self \
--pretrain logs/log_train_earlyaction_eucl_poincare_kinetics/20201025_122942/model/model_best_epoch9.pth.tar \
--path_dataset /proj/vondrick/datasets/FineGym \
--resume logs/log_finetune_earlyaction_self_eucl_poincare_finegym_lr3/20201103_115208/epoch32.pth.tar
