#!/usr/bin/env bash
# modified to lr4 after epoch 2
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9985 \
--nproc_per_node=4 \
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
--prefix earlyaction_trainall_finegym_kinetics_fromfinetune_lr3_dim2 \
--fp16 \
--fp64_hyper \
--pretrain logs/log_finetune_dpc_self_hyper_v1_poincare_finegym_finetuneto2d/20201102_110852/model/model_best_epoch193.pth.tar \
--linear_input predictions_z_hat \
--n_classes 4 \
--use_labels \
--num_workers 10 \
--early_action \
--num_workers 8 \
--seed 0 \
--action_level_gt \
--final_2dim \
--path_dataset /proj/vondrick/datasets/FineGym