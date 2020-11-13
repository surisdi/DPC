#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9997 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--network_feature resnet18 \
--dataset hollywood2 \
--seq_len 5 \
--num_seq 6 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--lr 0.001 \
--prefix finetune_earlyaction_euclidean_poincare_hollywood2_64d_fromhypermovienet \
--cross_gpu_score \
--early_action \
--early_action_self \
--feature_dim 64 \
--pretrain logs/log_train_dpc_movienet_hyperv1_64d/20201109_114711/model/model_best_epoch6.pth.tar \
--path_dataset /local/vondrick/didacsuris/local_data/Hollywood2
# --pretrain logs/log_train_dpc_movienet_hyperv1_64d_euclidean/20201109_155336/model/model_best_epoch3.pth.tar \

