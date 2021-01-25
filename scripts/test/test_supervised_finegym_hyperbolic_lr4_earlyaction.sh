#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9997 \
  --nproc_per_node=4 \
  main.py \
  --network_feature resnet18 \
  --dataset finegym \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 100 \
  --hyperbolic \
  --hyperbolic_version 1 \
  --pred_step 0 \
  --seq_len 5 \
  --num_seq 6 \
  --distance 'squared' \
  --lr 1e-4 \
  --prefix supervised_finegym_hyperbolic_lr4_earlyaction \
  --fp16 \
  --fp64_hyper \
  --pretrain /proj/vondrick/shared/hypvideo/logs/log_supervised_finegym_hyperbolic_lr4_earlyaction/20210125_115628/model/model_best_epoch30.pth.tar \
  --linear_input predictions_z_hat \
  --n_classes 307 \
  --action_level_gt \
  --use_labels \
  --num_workers 12 \
  --early_action \
  --num_workers 8 \
  --seed 0 \
  --path_dataset /local/vondrick/didacsuris/local_data/FineGym/ \
  --path_data_info /proj/vondrick/shared/hypvideo/dataset_info \
  --path_logs /proj/vondrick/shared/hypvideo/logs/ \
  --test
