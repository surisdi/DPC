#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9988 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 128 \
--img_dim 128 \
--epochs 100 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--lr 1e-3 \
--prefix test_earlyaction_linear_finegym_kinetics_fromfinetune_euclidean_lr3 \
--fp16 \
--pretrain logs/log_earlyaction_linear_finegym_kinetics_fromfinetune_euclidean_lr3/20201101_192504/model/model_best_epoch70.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--use_labels \
--num_workers 15 \
--only_train_linear \
--early_action \
--num_workers 8 \
--seed 0 \
--action_level_gt \
--test \
--path_dataset /proj/vondrick/datasets/FineGym