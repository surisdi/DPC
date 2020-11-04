#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9993 \
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
--epochs 200 \
--fp16 \
--fp64_hyper \
--num_workers 7 \
--lr 0.001 \
--prefix train_dpc_hyper_v1_cone_kinetics \
--hyp_cone \
--path_dataset /local/vondrick/ruoshi/k600 \
--margin 2 \
--resume logs/log_train_dpc_hyper_v1_cone_kinetics/20201031_202443/model/model_best_epoch10.pth.tar
