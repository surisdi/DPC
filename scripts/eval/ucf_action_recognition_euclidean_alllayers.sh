#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9998 \
  --nproc_per_node=4 \
  main.py \
--network_feature resnet18 \
--dataset ucf \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--lr 0.001 \
--prefix ucf_action_recognition_euclidean_alllayers \
--fp16 \
--pretrain /proj/vondrick/shared/hypvideo/logs/log_train_earlyaction_eucl_poincare_kinetics/20201025_122942/model/model_best_epoch9.pth.tar \
--linear_input predictions_z_hat \
--n_classes 101 \
--use_labels \
--num_workers 12 \
--seed 0 \
--path_dataset /proj/vondrick/datasets/UCF-101 \
--action_level_gt