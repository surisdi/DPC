#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 03
CUDA_VISIBLE_DEVICES=0 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9986 \
--nproc_per_node=1 \
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
--prefix test_future_subaction_linear_finegym_kinetics \
--fp16 \
--fp64_hyper \
--pretrain logs/log_future_subaction_linear_finegym_kinetics_fromfinetune_lr2/20201031_182603/model/model_best_epoch46.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--only_train_linear \
--pred_future \
--test \
--num_workers 8 \
--seed 0 \
--path_dataset /proj/vondrick/datasets/FineGym \
--test_info extract_features