import argparse
import os
import random
import re
import time
from datetime import datetime

import warnings
warnings.simplefilter("ignore", UserWarning)

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed
import torchvision.utils as vutils
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

import datasets
import models
from trainer import Trainer
from utils import augmentation
from utils.utils import AverageMeter, denorm, calc_topk_accuracy, neq_load_customized

plt.switch_backend('agg')

torch.backends.cudnn.benchmark = True


def print_r(args, text):
    """ Print only when the local rank is <=0 (only once)"""
    if args.local_rank <= 0:
        print(text)


def get_args():
    parser = argparse.ArgumentParser()
    # Task definition
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--hyperbolic', action='store_true', help='Hyperbolic mode')
    parser.add_argument('--hyperbolic_version', default=1, type=int)
    parser.add_argument('--distance', type=str, default='regular', help='Operation on top of the distance (hyperbolic)')
    parser.add_argument('--hyp_cone', action='store_true', help='Hyperbolic cone')
    parser.add_argument('--margin', default=0.01, type=float, help='margin for entailment cone loss')
    parser.add_argument('--early_action', action='store_true', help='Train with early action recognition loss')
    parser.add_argument('--early_action_self', action='store_true',
                        help='Only applies when early_action. Train without labels')
    parser.add_argument('--finetune', action='store_true', help='Finetune model')

    # Network
    parser.add_argument('--network_feature', default='resnet18', type=str, help='Network to use for feature extraction')
    # Data
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--n_classes', default=0, type=int)
    # Optimization
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    # Other
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--finetune_path', default='', type=str, help='path of pretrained model')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--debug', action='store_true', help='Debug. Do not store results')
    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for initialization')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training on gpus')
    parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit float precision instead of 32-bit. '
                                                            'Only affects the Euclidean layers')
    parser.add_argument('--fp64_hyper', action='store_true', help='Whether to use 64-bit float precision instead of '
                                                                  '32-bit for the hyperbolic layers and operations,'
                                                                  'Can be combined with --fp16')
    parser.add_argument('--cross_gpu_score', action='store_true',
                        help='Compute the score matrix using as negatives samples from different GPUs')

    args = parser.parse_args()

    if args.early_action_self:
        assert args.early_action, 'Read the explanation'
        assert args.pred_step == 1, 'We only want to predict the last one'
    elif args.early_action or args.finetune:
        assert args.pred_step == 0, 'We want to predict a label, not a feature'

    assert not (args.hyp_cone and not args.hyperbolic), 'Hyperbolic cone only works in hyperbolic mode'

    return args


def main():
    args = get_args()

    # Fix randomness
    seed = args.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = args.step_n_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.step_n_gpus = torch.distributed.get_world_size()
        
    # ---------------------------- Prepare model ----------------------------- #
    if args.local_rank <= 0:
        print_r(args, 'Preparing model')

    model = models.Model(args)
    model = model.to(args.device)

    params = model.parameters()
    optimizer = geoopt.optim.RiemannianAdam(params, lr=args.lr, weight_decay=args.wd, stabilize=10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 150], gamma=0.1)

    args.old_lr = None

    best_acc = 0
    iteration = 0

    # --- restart training --- #
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume.split('/')[-3]).group(1))
            print_r(args, f"=> loading resumed checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print_r(args, f'==== Change lr from {args.old_lor} to {args.lr} ====')
            print_r(args, f"=> loaded resumed checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print_r(args, f"[Warning] no checkpoint found at '{args.resume}'")

    if args.finetune:
        if os.path.isfile(args.finetune_path):
            print_r(args, f"=> loading pretrained checkpoint '{args.finetune_path}'")
            checkpoint = torch.load(args.finetune_path, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'], parts=['backbone', 'agg', 'network_pred'])
            print_r(args, f"=> loaded pretrained checkpoint '{args.finetune_path}' (epoch {checkpoint['epoch']})")
        else:
            print_r(args, f"=> no checkpoint found at '{args.finetune_path}'")

        for name, param in model.module.named_parameters():
            if name != 'network_class':
                param.requires_grad = False

    if args.local_rank != -1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        args.parallel = 'ddp'
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        args.parallel = 'dp'
    else:
        args.parallel = 'none'

    # ---------------------------- Prepare dataset ----------------------------- #
    train_loader = get_data(args, 'train', return_label=args.n_classes > 0)
    val_loader = get_data(args, 'val', return_label=args.n_classes > 0)

    # setup tools
    img_path, model_path = set_path(args)
    writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val') if not args.debug else '/tmp') if args.local_rank <= 0 else None
    writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train') if not args.debug else '/tmp') if args.local_rank <= 0 else None

    # ---------------------------- Prepare trainer and run ----------------------------- #
    if args.local_rank <= 0:
        print('Preparing trainer')
    trainer = Trainer(args, model, optimizer, train_loader, val_loader, iteration, best_acc, writer_train, writer_val,
                      img_path, model_path, scheduler, partial=0.1)
    trainer.train()


def get_data(args, mode='train', return_label=False):
    transform = None
    if args.dataset == 'ucf101':  # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            augmentation.RandomHorizontalFlip(consistent=True),
            augmentation.RandomCrop(size=224, consistent=True),
            augmentation.Scale(size=(args.img_dim, args.img_dim)),
            augmentation.RandomGray(consistent=False, p=0.5),
            augmentation.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            augmentation.ToTensor(),
            augmentation.Normalize()
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args.dataset == 'k400' or args.dataset == 'k600' or args.dataset == 'hollywood2':  # TODO think augmentation for hollywood2
        transform = transforms.Compose([
            augmentation.RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            augmentation.RandomHorizontalFlip(consistent=True),
            augmentation.RandomGray(consistent=False, p=0.5),
            augmentation.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            augmentation.ToTensor(),
            augmentation.Normalize()
        ])

    print_r(args, 'Loading data for "%s" ...' % mode)

    if args.dataset == 'k600':
        use_big_K600 = args.img_dim > 140
        dataset = datasets.Kinetics600_full_3d(mode=mode,
                                               transform=transform,
                                               seq_len=args.seq_len,
                                               num_seq=args.num_seq,
                                               downsample=5,
                                               big=use_big_K600,
                                               return_label=return_label)
    elif args.dataset == 'ucf101':
        dataset = datasets.UCF101_3d(mode=mode,
                                     transform=transform,
                                     seq_len=args.seq_len,
                                     num_seq=args.num_seq,
                                     downsample=args.ds,
                                     return_label=return_label)
    elif args.dataset == 'hollywood2':
        dataset = datasets.Hollywood2(mode=mode,
                                      transform=transform,
                                      seq_len=args.seq_len,
                                      num_seq=args.num_seq,
                                      downsample=args.ds,
                                      return_label=return_label)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    else:  # mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print_r(args, '"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_path = f"logs/log_{args.prefix}/{current_time}"
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if args.local_rank <= 0:
        os.makedirs(img_path)
        os.makedirs(model_path)
    return img_path, model_path


if __name__ == '__main__':
    main()
