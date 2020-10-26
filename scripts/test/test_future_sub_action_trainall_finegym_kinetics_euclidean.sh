#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 01
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9993 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 80 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--prefix test_future_sub_action_linear_finegym_kinetics_euclidean \
--fp16 \
--pretrain logs/log_future_sub_action_trainall_finegym_kinetics_euclidean/20201022_150239/model/model_best_epoch99.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--test \
--verbose
#--resume logs/log_future_sub_action_trainall_finegym_kinetics_euclidean/20201022_150239/model/model_best_epoch99.pth.tar
