#!/usr/bin/env bash
# Aquest te bons re
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9984 \
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
--prefix future_subaction_linear_finegym_kinetics_64d \
--fp16 \
--pretrain logs/log_finetune_dpc_eucl_poincare_finegym_64d/20201109_200623/model/model_best_epoch97.pth.tar \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 12 \
--only_train_linear \
--pred_future \
--num_workers 8 \
--seed 0 \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym \
--feature_dim 64 \
--resume logs/log_future_subaction_linear_finegym_kinetics_fromfinetune_euclidean_lr3_model2/20201104_224751/model/epoch91.pth.tar

