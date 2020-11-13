#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9986 \
--nproc_per_node=3 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--lr 1e-2 \
--prefix earlyaction_linear_finegym_64d_euclidean \
--fp16 \
--pretrain logs/log_finetune_earlyaction_self_eucl_poincare_finegym_64d/20201109_201936/model/model_best_epoch104.pth.tar  \
--linear_input predictions_z_hat \
--n_classes 307 \
--use_labels \
--num_workers 15 \
--only_train_linear \
--early_action \
--num_workers 8 \
--seed 0 \
--action_level_gt \
--feature_dim 64 \
--path_dataset /proj/vondrick/datasets/FineGym

