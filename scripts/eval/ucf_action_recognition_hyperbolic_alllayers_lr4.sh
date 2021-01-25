#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9999 \
  --nproc_per_node=4 \
  main.py \
--network_feature resnet18 \
--dataset ucf \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--pred_step 0 \
--seq_len 5 \
--num_seq 8 \
--lr 0.0001 \
--prefix ucf_action_recognition_hyperbolic_alllayers_lr4 \
--fp16 \
--pretrain /proj/vondrick/shared/hypvideo/logs/log_finetune_earlyaction_self_hyper_v1_poincare_finegym/20201025_164701/model/model_best_epoch114.pth.tar \
--linear_input predictions_c \
--n_classes 101 \
--use_labels \
--num_workers 12 \
--seed 0 \
--path_dataset /proj/vondrick/datasets/UCF-101 \
--action_level_gt \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'squared' \
--fp64_hyper \
--resume /proj/vondrick/didac/code/DPC/logs/log_ucf_action_recognition_hyperbolic_alllayers_lr4/20210123_104612/model/model_best_epoch8.pth.tar

