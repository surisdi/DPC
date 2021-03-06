#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 03
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9990 \
--nproc_per_node=4 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--lr 1e-3 \
--prefix earlyaction_correct_linear_finegym_kinetics_lr3_fromfinetune_euclidean \
--fp16 \
--pretrain logs/log_finetune_dpc_eucl_poincare_finegym/20201029_185002/model/model_best_epoch135.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--use_labels \
--num_workers 12 \
--only_train_linear \
--action_level_gt \
--early_action
#--cross_gpu_score  Only needed when training with contrastive loss
