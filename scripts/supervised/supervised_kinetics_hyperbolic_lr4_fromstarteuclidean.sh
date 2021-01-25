#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9997 \
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
  --lr 1e-4 \
  --prefix supervised_kinetics_hyperbolic_lr4_fromstarteuclidean \
  --action_level_gt \
  --fp16 \
  --linear_input predictions_c \
  --n_classes 600 \
  --use_labels \
  --num_workers 10 \
  --seed 0 \
  --path_dataset /local/vondrick/didacsuris/local_data/kinetics-600/data/extracted_frames/ \
  --path_data_info /proj/vondrick/shared/hypvideo/dataset_info \
  --path_logs /proj/vondrick/shared/hypvideo/logs/ \
  --hyperbolic \
  --hyperbolic_version 1 \
  --distance 'squared' \
  --fp64_hyper \
  --resume /proj/vondrick/shared/hypvideo/logs/log_supervised_kinetics_hyperbolic_lr4_fromstarteuclidean/20210120_173946/model/model_best_epoch6_.pth.tar \
  --pretrain /proj/vondrick/shared/hypvideo/logs/log_supervised_kinetics_euclidean/20210120_121622/model/model_best_epoch10_.pth.tar
