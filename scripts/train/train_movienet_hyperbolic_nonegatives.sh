#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9999 \
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
  --ds 3 \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 200 \
  --fp16 \
  --fp64_hyper \
  --num_workers 15 \
  --lr 0.0001 \
  --prefix train_movienet_hyperbolic_nonegatives \
  --path_dataset /local/vondrick/didacsuris/local_data/MovieNet \
  --path_data_info /proj/vondrick/shared/hypvideo/dataset_info \
  --no_hard_negs
