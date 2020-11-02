import torch
import numpy as np
import sys
sys.path.append('../')
import models
import datasets
from tqdm import tqdm
from utils.hyp_cone_utils import cone_distance_sum
from utils.poincare_distance import poincare_distance
from losses import compute_mask

from datasets import Kinetics600_full_3d
from torchvision import transforms
from utils import augmentation
from torch.utils import data
import pickle


import os
import argparse

def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # deterministic random dataloader
    torch.manual_seed(2020)
    torch.backends.cudnn.deterministic = True
    
    # initialize model
    model_path = args.pretrain
    model = models.Model(args)
    model = model.to(args.device)
    
    # load model
    checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = torch.nn.DataParallel(model)
    model.eval()
    
    # dataloader
    dataloader = datasets.get_data(args, 'test', return_label=args.use_labels,\
                                   hierarchical_label=args.hierarchical_labels,\
                                   action_level_gt=args.action_level_gt,\
                                   num_workers=args.num_workers, \
                                   vis=True)
    
    # collect features
    all_features = []
    all_preds = []
    all_vpaths = []
    all_idx_blocks = []
    
    print('\n==== Generating embeddings for pretrain model ====\n')
    with torch.no_grad():
        for idx, (input_dict, label) in tqdm(enumerate(dataloader), total=int(len(dataloader) * args.partial)):
            input_seq = input_dict['t_seq'].cuda()
            pred, feature_dist, sizes = model(input_seq)
            target, (B, B2, NS, NP, SQ) = compute_mask(args, sizes, args.batch_size)
            _, D = pred.shape
            pred = pred.reshape(B, NP, SQ, D)
            feature_dist = feature_dist.reshape(B, NS, SQ, D)
            pred_pooled = torch.mean(pred, dim=2).reshape(-1, D)
            feature_dist_pooled = torch.mean(feature_dist, dim=2).reshape(-1, D)
            del pred, feature_dist
            pred_pooled = pred_pooled.reshape(B, NP, D)
            feature_dist_pooled = feature_dist_pooled.reshape(B, NS, D)
            all_features.append(feature_dist_pooled.cpu().detach())
            all_preds.append(pred_pooled.cpu().detach())
            all_vpaths.extend(input_dict['vpath'])
            all_idx_blocks.append(input_dict['idx_block'])
            del input_dict
            if idx >= int(len(dataloader) * args.partial):
                break
        
    all_features = torch.cat(all_features)
    all_preds = torch.cat(all_preds)
    all_idx_blocks = torch.cat(all_idx_blocks)
    
    features_info = {'feature': all_features, 'pred': all_preds, 'vpath': all_vpaths, 'idx_block': all_idx_blocks}
    
    print('\n==== Saving features... ====\n')
    
    model_path = args.pretrain
    base_path = '/'.join(model_path.split('/')[:-2])
    embedding_path = os.path.join(base_path, 'embeds')
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)

    f = open(os.path.join(embedding_path, model_path.split('/')[-1][:-8] + '_embeds.pkl'),'wb')
    pickle.dump(features_info,f)
    f.close()




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    # Model
    parser.add_argument('--hyperbolic', action='store_true', help='Hyperbolic mode')
    parser.add_argument('--hyperbolic_version', default=1, type=int)
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str,
                        help='path of pretrained model. Difference with resume is that we start a completely new '
                             'training and checkpoint, do not load optimizer, and model loading is not strict')
    parser.add_argument('--only_train_linear', action='store_true',
                        help='Only train last linear layer. Only used (only makes sense) if pretrain is used.')
    parser.add_argument('--linear_input', default='features', type=str, help='Input to the last linear layer',
                        choices=['features_z', 'predictions_c', 'predictions_z_hat'])
    parser.add_argument('--network_feature', default='resnet18', type=str, help='Network to use for feature extraction')
    # Loss
    parser.add_argument('--distance', type=str, default='regular', help='Operation on top of the distance (hyperbolic)')
    parser.add_argument('--hyp_cone', action='store_true', help='Hyperbolic cone')
    parser.add_argument('--margin', default=0.1, type=float, help='margin for entailment cone loss')
    parser.add_argument('--early_action', action='store_true', help='Train with early action recognition loss')
    parser.add_argument('--early_action_self', action='store_true',
                        help='Only applies when early_action. Train without labels')
    parser.add_argument('--pred_step', default=3, type=int, help='How subclips to predict')
    parser.add_argument('--cross_gpu_score', action='store_false',
                        help='Compute the score matrix using as negatives samples from different GPUs')
    parser.add_argument('--hierarchical_labels', action='store_true',
                        help='Works both for training with labels and for testing the accuracy')
    parser.add_argument('--test', action='store_true', help='Test system')
    # Data
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--seq_len', default=5, type=int, help='Number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='Number of video blocks')
    parser.add_argument('--ds', default=3, type=int, help='Frame downsampling rate')
    parser.add_argument('--n_classes', default=0, type=int)
    parser.add_argument('--use_labels', action='store_true', help='Return labels in dataset and use supervised loss')
    parser.add_argument('--action_level_gt', action='store_true',
                        help='As opposed to subaction level. If True, we do not evaluate subactions or hierarchies')
    parser.add_argument('--img_dim', default=128, type=int)
    # Training
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='Number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch number (useful on restarts)')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--partial', default=1., type=float, help='Percentage of training set to use')
    # Other
    parser.add_argument('--print_freq', default=5, type=int, help='Frequency of printing output during training')
    parser.add_argument('--verbose', action='store_true', help='Print information')
    parser.add_argument('--debug', action='store_true', help='Debug. Do not store results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on gpus')
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit float precision instead of 32-bit. '
                                                            'Only affects the Euclidean layers')
    parser.add_argument('--fp64_hyper', action='store_true', help='Whether to use 64-bit float precision instead of '
                                                                  '32-bit for the hyperbolic layers and operations,'
                                                                  'Can be combined with --fp16')
    parser.add_argument('--num_workers', default=32, type=int, help='number of workers for dataloader')

    args = parser.parse_args()

    if args.early_action_self:
        assert args.early_action, 'Read the explanation'
        assert args.pred_step == 1, 'We only want to predict the last one'

    if args.use_labels:
        assert args.pred_step == 0, 'We want to predict a label, not a feature'

    assert not (args.hyp_cone and not args.hyperbolic), 'Hyperbolic cone only works in hyperbolic mode'

    if args.early_action and not args.early_action_self:
        assert args.use_labels
        assert args.action_level_gt, 'Early action recognition implies only action level, not subaction level'

    if args.action_level_gt:
        assert args.linear_input != 'features_z', 'We cannot get a representation for the whole clip with features_z'

    return args




def compute_mask(args, sizes, B):
    if args.use_labels:
        return None, None  # No need to compute mask

    last_size, size_gt, size_pred = sizes[0]

    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    mask = torch.zeros((B, size_pred, last_size ** 2, B, size_gt, last_size ** 2), dtype=torch.int8, requires_grad=False).detach().cuda()

    mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg

    if args.early_action_self:
        pass  # Here NO temporal neg! All steps try to predict the last one
    else:
        for k in range(B):
            mask[k, :, torch.arange(last_size ** 2), k, :, torch.arange(last_size ** 2)] = -1  # temporal neg

    tmp = mask.permute(0,2,1,3,5,4).reshape(B * last_size ** 2, size_pred, B * last_size ** 2, size_gt)

    if args.early_action_self:
        tmp[torch.arange(B * last_size ** 2), :, torch.arange(B * last_size ** 2)] = 1  # pos
    else:
        assert size_gt == size_pred
        for j in range(B * last_size ** 2):
            tmp[j, torch.arange(size_pred), j, torch.arange(size_gt)] = 1  # pos

    mask = tmp.view(B, last_size ** 2, size_pred, B, last_size ** 2, size_gt).permute(0, 2, 1, 3, 5, 4)

    # Now, given task mask as input, compute the target for contrastive loss
    if mask is None:
        return None, None
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


if __name__ == '__main__':
    main()


