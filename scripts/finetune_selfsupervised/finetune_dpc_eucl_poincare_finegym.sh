#!/usr/bin/env bash
# Here just reusing the fact that we trained an earlyaction one and not start from scratch
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--network_feature resnet18 \
--dataset finegym \
--seq_len 5 \
--num_seq 6 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--num_workers 15 \
--lr 0.0001 \
--prefix finetune_dpc_eucl_poincare_finegym \
--cross_gpu_score \
--pretrain logs/log_train_earlyaction_eucl_poincare_kinetics/20201025_122942/model/model_best_epoch9.pth.tar \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym/
