#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 01
CUDA_VISIBLE_DEVICES=4 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9987 \
--nproc_per_node=1 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--lr 1e-3 \
--prefix test_future_subaction_trainall_finegym_kinetics_euclidean \
--fp16 \
--pretrain logs/log_future_subaction_linear_finegym_kinetics_fromfinetune_euclidean_lr3/20201031_204139/model/model_best_epoch58.pth.tar  \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--only_train_linear \
--pred_future \
--test \
--num_workers 8 \
--seed 1 \
--path_dataset /proj/vondrick/datasets/FineGym
