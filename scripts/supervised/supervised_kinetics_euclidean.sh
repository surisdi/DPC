#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9999 \
  --nproc_per_node=4 \
  main.py \
  --network_feature resnet18 \
  --dataset kinetics \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 100 \
  --pred_step 0 \
  --seq_len 5 \
  --num_seq 8 \
  --lr 1e-3 \
  --prefix supervised_kinetics_euclidean \
  --action_level_gt \
  --fp16 \
  --linear_input predictions_c \
  --n_classes 600 \
  --use_labels \
  --num_workers 32 \
  --seed 0 \
  --path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data/extracted_frames/ \
  --path_data_info /proj/vondrick/shared/hypvideo/dataset_info \
  --path_logs /proj/vondrick/shared/hypvideo/logs/ \
  --partial 0.1