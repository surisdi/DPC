#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 01
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9995 \
--nproc_per_node=2 \
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
--prefix future_sub_action_trainall_finegym_kinetics_euclidean_fromfinetune \
--fp16 \
--pretrain logs/log_finetune_earlyaction_self_eucl_poincare_finegym/20201026_105155/model/model_best_epoch133.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
#--cross_gpu_score  Only needed when training with contrastive loss
