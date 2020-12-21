#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
  -W ignore \
  -i \
  -m torch.distributed.launch \
  --master_port=9999 \
  --nproc_per_node=4 \
  main.py \
  --network_feature resnet18 \
  --dataset finegym \
  --batch_size 32 \
  --img_dim 128 \
  --epochs 100 \
  --pred_step 0 \
  --seq_len 5 \
  --num_seq 6 \
  --lr 1e-3 \
  --prefix future_subaction_linear_finegym_fromfinetune_euclidean \
  --fp16 \
  --pretrain /path/to/model/pretrained/on/finegym/checkpoint.pth.tar \
  --linear_input predictions_z_hat \
  --n_classes 307 \
  --hierarchical_labels \
  --use_labels \
  --num_workers 12 \
  --only_train_linear \
  --pred_future \
  --num_workers 8 \
  --seed 0 \
  --path_dataset /path/to/datasets/FineGym
