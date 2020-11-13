
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_JIT=0 NCCL_LL_THRESHOLD=0 python \
-W ignore \
-i \
-m torch.distributed.launch \
--master_port=9987 \
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
--prefix future_subaction_finegym_predictlevel_trainall \
--fp16 \
--fp64_hyper \
--linear_input predictions_z_hat \
--n_classes 307 \
--hierarchical_labels \
--use_labels \
--num_workers 15 \
--pred_future \
--predict_level \
--num_workers 12 \
--seed 0 \
--path_dataset /local/vondrick/didacsuris/local_data/FineGym \
--pretrain logs/log_future_subaction_finegym_predictlevel_trainall/20201109_182917/model/model_best_epoch1.pth.tar
#--pretrain logs/log_future_subaction_linear_finegym_kinetics_fromfinetune_lr2/20201031_182603/model/model_best_epoch46.pth.tar
#--pretrain logs/log_finetune_dpc_self_hyper_v1_poincare_finegym_fromearlyaction/20201028_122531/model/model_best_epoch7.pth.tar \
#--pretrain logs/log_finetune_earlyaction_self_hyper_v1_poincare_finegym/20201025_164701/model/model_best_epoch114.pth.tar \


