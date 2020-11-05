#!/usr/bin/env bash
# Instead of finetuning from the same version in multiple dimensions, we finetune from dpc in 2d. Probably faster
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9998 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--hyperbolic \
--hyperbolic_version 1 \
--distance squared \
--network_feature resnet18 \
--dataset finegym \
--seq_len 5 \
--num_seq 6 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 15 \
--lr 0.0001 \
--prefix finetune_earlyaction_self_hyper_v1_poincare_finegym_2dim \
--cross_gpu_score \
--early_action \
--early_action_self \
--pretrain logs/log_finetune_dpc_self_hyper_v1_poincare_finegym_finetuneto2d/20201102_110852/model/model_best_epoch193.pth.tar \
--final_2dim \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym