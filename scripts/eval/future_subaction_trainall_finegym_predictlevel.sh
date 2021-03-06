#!/usr/bin/env bash
# We want prediction a la early action, but for a subaction. This is why we put early_action
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
--prefix future_subaction_finegym_predictlevel_trainall \
--fp16 \
--fp64_hyper \
--linear_input predictions_z_hat \
--n_classes 310 \
--use_labels \
--num_workers 15 \
--early_action \
--predict_level \
--num_workers 8 \
--seed 0 \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym \
--hierarchical_labels \
--pretrain logs/log_finetune_dpc_self_hyper_v1_poincare_finegym_fromearlyaction/20201028_122531/model/model_best_epoch7.pth.tar
#--pretrain logs/log_finetune_earlyaction_self_hyper_v1_poincare_finegym/20201025_164701/model/model_best_epoch114.pth.tar \


