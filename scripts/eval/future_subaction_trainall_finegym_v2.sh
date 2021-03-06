#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9974 \
--nproc_per_node=4 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--hyperbolic \
--hyperbolic_version 2 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--distance 'squared' \
--lr 1e-2 \
--prefix future_subaction_trainall_finegym_v2 \
--fp16 \
--fp64_hyper \
--pretrain logs/log_finetune_dpc_self_hyper_v2_poincare_finegym/20201103_192241/model/model_best_epoch192.pth.tar  \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 15 \
--pred_future \
--num_workers 8 \
--seed 0 \
--hyp_cone \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym \
--not_track_running_stats

