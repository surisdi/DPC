#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9989 \
  --nproc_per_node=3 \
  main.py \
  --network_feature resnet18 \
  --dataset hollywood2 \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 100 \
  --hyperbolic \
  --hyperbolic_version 1 \
  --pred_step 0 \
  --seq_len 5 \
  --num_seq 6 \
  --distance 'squared' \
  --lr 1e-2 \
  --prefix earlyaction_linear_hollywood2_fromfinetune_hyperbolic_onc \
  --fp16 \
  --fp64_hyper \
  --pretrain /proj/vondrick/shared/hypvideo/logs/log_earlyaction_linear_hollywood2_fromfinetune_hyperbolic_onc/20210125_101133/model/model_best_epoch86.pth.tar \
  --linear_input predictions_c \
  --n_classes 17 \
  --use_labels \
  --num_workers 12 \
  --only_train_linear \
  --early_action \
  --num_workers 8 \
  --seed 0 \
  --hierarchical_labels \
  --path_dataset /local/vondrick/didacsuris/Hollywood2 \
  --path_data_info /proj/vondrick/shared/hypvideo/dataset_info \
  --path_logs /proj/vondrick/shared/hypvideo/logs/ \
  --action_level_gt \
  --test