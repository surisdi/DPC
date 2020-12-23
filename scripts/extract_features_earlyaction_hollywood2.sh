#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9996 \
--nproc_per_node=2 \
main.py \
--network_feature resnet18 \
--dataset hollywood2 \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--hyperbolic \
--hyperbolic_version 1 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--distance 'squared' \
--lr 1e-2 \
--prefix earlyaction_linear_hollywood2_movienet_fromfinetune_lr2 \
--fp16 \
--fp64_hyper \
--pretrain /proj/vondrick/shared/hypvideo/logs/log_earlyaction_linear_hollywood2_movienet_fromfinetune_lr2/20201106_191448/model/model_best_epoch47.pth.tar \
--linear_input predictions_z_hat \
--n_classes 17 \
--use_labels \
--num_workers 15 \
--only_train_linear \
--early_action \
--num_workers 8 \
--seed 0 \
--action_level_gt \
--hierarchical_labels \
--path_dataset /local/vondrick/didacsuris/Hollywood2 \
--not_track_running_stats \
--test \
--test_info extract_features

