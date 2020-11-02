#!/usr/bin/env bash
# This is the number 03 in the list
# Changed from lr3 to lr4 in epoch 13 (also started epoch 1 with only 1 gpu)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--pred_step 3 \
--hyperbolic \
--hyperbolic_version 1 \
--distance squared \
--network_feature resnet18 \
--dataset movienet \
--seq_len 3 \
--num_seq 8 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 10 \
--lr 0.0001 \
--prefix train_dpc_hyper_v1_poincare_movienet \
--path_dataset /local/vondrick/didacsuris/local_data/MovieNet \
--resume logs/log_train_dpc_hyper_v1_poincare_movienet/20201101_130308/model/model_best_epoch14.pth.tar
