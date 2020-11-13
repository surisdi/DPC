#!/usr/bin/env bash
# started training from epoch 1 lowering lr to 0.0001
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9997 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--hyperbolic \
--hyperbolic_version 1 \
--distance squared \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 8 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 100 \
--fp16 \
--fp64_hyper \
--num_workers 15 \
--lr 0.0001 \
--prefix train_dpc_kinetics_hyperv1_8d \
--feature_dim 8 \
--cross_gpu_score \
--path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data \
--seed 0 \
--pretrain logs/log_train_earlyaction_hyper_v1_poincare_kinetics_lr4/20201023_151021/model/model_best_epoch31.pth.tar \
--resume logs/log_train_dpc_kinetics_hyperv1_8d/20201107_172536/model/model_best_epoch1.pth.tar 
