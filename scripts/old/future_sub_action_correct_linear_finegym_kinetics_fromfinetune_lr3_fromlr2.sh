#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 03
CUDA_VISIBLE_DEVICES=0 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9993 \
--nproc_per_node=1 \
main.py \
--network_feature resnet18 \
--dataset finegym \
--batch_size 32 \
--img_dim 128 \
--epochs 118 \
--hyperbolic \
--hyperbolic_version 1 \
--pred_step 0 \
--seq_len 5 \
--num_seq 6 \
--distance 'squared' \
--lr 1e-3 \
--prefix future_sub_action_correct_linear_finegym_kinetics_lr3_fromlr2 \
--fp16 \
--fp64_hyper \
--pretrain logs/log_future_sub_action_correct_linear_finegym_kinetics_lr2_fromfinetune/20201028_132820/model/model_best_epoch32.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--only_train_linear \
--pred_future \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym \
--resume logs/log_future_sub_action_correct_linear_finegym_kinetics_lr3_fromlr2/20201029_101437/model/epoch117_.pth.tar
#--cross_gpu_score  Only needed when training with contrastive loss
