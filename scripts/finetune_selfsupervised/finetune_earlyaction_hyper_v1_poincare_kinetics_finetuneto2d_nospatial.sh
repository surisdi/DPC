#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9997 \
--nproc_per_node=4 \
main.py \
--pred_step 1 \
--hyperbolic \
--hyperbolic_version 1 \
--distance squared \
--network_feature resnet18 \
--dataset k600 \
--seq_len 5 \
--num_seq 6 \
--ds 3 \
--batch_size 32 \
--img_dim 128 \
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 12 \
--lr 0.001 \
--prefix finetune_earlyaction_hyper_v1_poincare_kinetics_2dim_nospatial \
--cross_gpu_score \
--early_action \
--early_action_self \
--pretrain logs/log_train_earlyaction_hyper_v1_poincare_kinetics_lr4/20201023_151021/model/model_best_epoch31.pth.tar \
--final_2dim \
--no_spatial \
--path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data \
--partial 0.1 \
#--resume logs/log_finetune_earlyaction_hyper_v1_poincare_kinetics_2dim_nospatial/20201104_133510/model/model_best_epoch4.pth.tar
