#!/usr/bin/env bash
# In this example we implement the eval option 13, from training option 01
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9994 \
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
--prefix future_sub_action_correct_linear_finegym_kinetics_euclidean_fromfinetune \
--fp16 \
--pretrain logs/log_finetune_earlyaction_self_eucl_poincare_finegym/20201026_105155/model/model_best_epoch133.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--only_train_linear \
--pred_future \
--path_dataset /proj/vondrick/datasets/FineGym \
#--resume logs/log_future_sub_action_correct_linear_finegym_kinetics_euclidean_fromfinetune/20201028_133306/model/model_best_epoch3.pth.tar \
#--cross_gpu_score  Only needed when training with contrastive loss
