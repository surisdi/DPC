import os
import sys
import time
import re
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../utils')
from dataset_3d import *
from model_3d import *
from augmentation import *
from utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy, neq_load_customized
import geoopt

import torch
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import math

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--model', default='dpc-rnn', type=str)
parser.add_argument('--dataset', default='ucf101', type=str)
parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume')
parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--train_what', default='all', type=str)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--hyperbolic', action='store_true', help='Hyperbolic mode')
parser.add_argument('--hyperbolic_version', default=1, type=int)
parser.add_argument('--distance', type=str, default='regular', help='Operation on top of the distance (hyperbolic)')
parser.add_argument('--hyp_cone', action='store_true', help='Hyperbolic mode')
parser.add_argument('--margin', default=0.01, type=float, help='margin for entailment cone loss')

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    global args; args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    global cuda; cuda = torch.device('cuda')

    ### dpc model ###
    if args.model == 'dpc-rnn':
        model = DPC_RNN(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step,
                        hyperbolic=args.hyperbolic,
                        hyperbolic_version=args.hyperbolic_version,
                        distance = args.distance,
                        hyp_cone = args.hyp_cone,
                       )
    else: raise ValueError('wrong model!')

    model = nn.DataParallel(model)
    model = model.to(cuda)
    global criterion; criterion = nn.CrossEntropyLoss()

    ### optimizer ###
    if args.train_what == 'last':
        for name, param in model.module.resnet.named_parameters():
            param.requires_grad = False
    else: pass # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    params = model.parameters()
    # optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    optimizer = geoopt.optim.RiemannianAdam(params, lr=args.lr, weight_decay=args.wd, stabilize=10)
    args.old_lr = None

    best_acc = 0
    global iteration; iteration = 0

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            args.old_lr = float(re.search('_lr(.+?)_', args.resume).group(1))
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.reset_lr: # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint['optimizer'])
            else: print('==== Change lr from %f to %f ====' % (args.old_lr, args.lr))
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'], parts = ['backbone', 'agg', 'network_pred'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else: 
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    ### load data ###
    if args.dataset == 'ucf101': # designed for ucf101, short size=256, rand crop to 224x224 then scale to 128x128
        transform = transforms.Compose([
            RandomHorizontalFlip(consistent=True),
            RandomCrop(size=224, consistent=True),
            Scale(size=(args.img_dim,args.img_dim)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])
    # designed for kinetics400, short size=150, rand crop to 128x128
    elif args.dataset == 'k400' or args.dataset == 'k600' or args.dataset == 'hollywood2':  # TODO think augmentation for hollywood2
        transform = transforms.Compose([
            RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')

    # setup tools
    global de_normalize; de_normalize = denorm()
    global img_path; img_path, model_path = set_path(args)
    global writer_train
    try: # old version
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, 'train'))
    except: # v1.7
        writer_val = SummaryWriter(logdir=os.path.join(img_path, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(img_path, 'train'))
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        
        # save curve
        if args.hyperbolic and args.hyp_cone:
            
            train_loss, train_pos_acc, train_neg_acc, train_p_norm, train_g_norm = train(train_loader, model, optimizer, epoch)
            val_loss, val_pos_acc, val_neg_acc, val_p_norm, val_g_norm = validate(val_loader, model, epoch)
            
            writer_train.add_scalar('global/loss', train_loss, epoch)
            writer_val.add_scalar('global/loss', val_loss, epoch)
            writer_train.add_scalar('accuracy/pos', train_pos_acc, epoch) # positive accuracy
            writer_train.add_scalar('accuracy/neg', train_neg_acc, epoch) # negative accuracy
            writer_train.add_scalar('accuracy/pnorm', train_p_norm, epoch) # average norm of predicted embedding
            writer_train.add_scalar('accuracy/gnorm', train_g_norm, epoch) # average norm of ground truth embedding
            writer_val.add_scalar('accuracy/pos', val_pos_acc, epoch)
            writer_val.add_scalar('accuracy/neg', val_neg_acc, epoch)
            writer_val.add_scalar('accuracy/pnorm', val_p_norm, epoch)
            writer_val.add_scalar('accuracy/gnorm', val_g_norm, epoch)
        else:
            
            train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch)
            val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)
            
            writer_train.add_scalar('global/loss', train_loss, epoch)
            writer_train.add_scalar('global/accuracy', train_acc, epoch)
            writer_val.add_scalar('global/loss', val_loss, epoch)
            writer_val.add_scalar('global/accuracy', val_acc, epoch)
            writer_train.add_scalar('accuracy/top1', train_accuracy_list[0], epoch)
            writer_train.add_scalar('accuracy/top3', train_accuracy_list[1], epoch)
            writer_train.add_scalar('accuracy/top5', train_accuracy_list[2], epoch)
            writer_val.add_scalar('accuracy/top1', val_accuracy_list[0], epoch)
            writer_val.add_scalar('accuracy/top3', val_accuracy_list[1], epoch)
            writer_val.add_scalar('accuracy/top5', val_accuracy_list[2], epoch)
        
        # save check_point
        if args.hyperbolic and args.hyp_cone:
            is_best = val_pos_acc > best_acc; best_acc = max(val_pos_acc, best_acc)
        else:
            is_best = val_acc > best_acc; best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'net': args.net,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': iteration}, 
                         is_best, filename=os.path.join(model_path, 'epoch%s.pth.tar' % str(epoch+1)), keep_all=False)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    train_pos_acc = AverageMeter()
    train_neg_acc = AverageMeter()
    train_p_norm = AverageMeter()
    train_g_norm = AverageMeter()
    model.train()
    global iteration

    time_last = time.time()
    for idx, input_seq in enumerate(data_loader):

        a = time.time()
        time_data = a - time_last
        input_seq = input_seq.to(cuda)
        B = input_seq.size(0)
        # visualize
        if (iteration == 0) or (iteration == args.print_freq):
            if B > 2: input_seq = input_seq[0:2,:]
            writer_train.add_image('input_seq',
                                   de_normalize(vutils.make_grid(
                                       input_seq.transpose(2,3).contiguous().view(-1,3,args.img_dim,args.img_dim), 
                                       nrow=args.num_seq*args.seq_len)),
                                   iteration)
#         del input_seq

        b = time.time()
        with torch.autograd.set_detect_anomaly(True):
            if args.hyperbolic and args.hyp_cone:
                [score_, mask_, pred_norm, gt_norm] = model(input_seq)
#                 criterion = nn.MSELoss().double()
#                 score_flat = score_.flatten()
#                 target = torch.zeros_like(score_flat)
#                 loss = criterion(score_flat, target.clone())

                pred_norm = torch.mean(pred_norm)
                gt_norm = torch.mean(gt_norm)
                loss = score_.sum()

                [A, B] = score_.shape
                score_ = score_[:B, :]
                pos = score_.diagonal(dim1=-2, dim2=-1)
                pos_acc = float((pos == 0).sum().item()) / float(pos.flatten().shape[0])
                k = score_.shape[0]
                score_.as_strided([k], [k + 1]).copy_(torch.zeros(k));
                neg_acc = float((score_ == 0).sum().item() - k) / float(k ** 2 - k)
                train_pos_acc.update(pos_acc, B)
                train_neg_acc.update(neg_acc, B)
                train_p_norm.update(pred_norm.item())
                train_g_norm.update(gt_norm.item())
                losses.update(loss.item() / (2 * k**2), B)

            else:
                [score_, mask_] = model(input_seq)
                if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

                # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
                # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
                score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target_flattened.float().argmax(dim=1)

                loss = criterion(score_flattened, target_flattened)
                top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

                accuracy_list[0].update(top1.item(),  B)
                accuracy_list[1].update(top3.item(), B)
                accuracy_list[2].update(top5.item(), B)

                losses.update(loss.item(), B)
                accuracy.update(top1.item(), B)

            c = time.time()

#             del score_

            optimizer.zero_grad()
#             try:
            loss.backward()
#             except: # debugging entailment cone loss
#                 nan_score = 0
#                 for i in score_.flatten():
#                     if math.isnan(i):
#                         nan_score += 1
#                 print('number of nan value in score:', nan_score)

#                 pnorm = torch.norm(pred, p=2, dim=-1)
#                 gnorm = torch.norm(gt, p=2, dim=-1)
#                 pcount = 0
#                 for p in pnorm:
#                     if p < 0.1:
#                         pcount += 1
#                 gcount = 0
#                 for g in gnorm:
#                     if g < 0.1:
#                         gcount += 1
#                 print('pcount:', pcount)
#                 print('gcount:', gcount)
#                 sys.exit()
            optimizer.step()

        del loss

        if idx % args.print_freq == 0:
            if args.hyperbolic and args.hyp_cone:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                      'Acc: pos {3:.4f}; neg {4:.4f}; pnorm {5:.4f}; gnorm {6:.4f}; T:{7:.2f} TD:{8:.2f}\t'.format(
                       epoch, idx, len(data_loader), pos_acc, neg_acc, pred_norm, gt_norm, time.time()-a, time_data, loss=losses))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.6f} ({loss.local_avg:.4f})\t'
                      'Acc: top1 {3:.4f}; top3 {4:.4f}; top5 {5:.4f} T:{6:.2f} TD:{7:.2f}\t'.format(
                       epoch, idx, len(data_loader), top1, top3, top5, time.time()-a, time_data, loss=losses))

            writer_train.add_scalar('local/loss', losses.val, iteration)
            writer_train.add_scalar('local/accuracy', accuracy.val, iteration)

            iteration += 1

        d = time.time()

        # print(time_data, b-a, c-b, d-c)

        time_last = time.time()

    # return different trainign statistics for different objective
    if args.hyperbolic and args.hyp_cone:
        return losses.local_avg, train_pos_acc.local_avg, train_neg_acc.local_avg, train_p_norm.local_avg, train_g_norm.local_avg
    else:
        return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    val_pos_acc = AverageMeter()
    val_neg_acc = AverageMeter()
    val_p_norm = AverageMeter()
    val_g_norm = AverageMeter()
    model.eval()

    with torch.no_grad():
        for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(cuda)
            B = input_seq.size(0)
#             del input_seq

            if args.hyperbolic and args.hyp_cone:
                [score_, mask_, pred_norm, gt_norm] = model(input_seq)
                pred_norm = torch.mean(pred_norm)
                gt_norm = torch.mean(gt_norm)
                loss = score_.sum()

                [A, B] = score_.shape
                score_ = score_[:B, :]
                pos = score_.diagonal(dim1=-2, dim2=-1)
                pos_acc = float((pos == 0).sum().item()) / float(pos.flatten().shape[0])
                k = score_.shape[0]
                score_.as_strided([k], [k + 1]).copy_(torch.zeros(k));
                neg_acc = float((score_ == 0).sum().item() - k) / float(k ** 2 - k)
                accuracy_list[0].update(pos_acc, B)
                accuracy_list[1].update(neg_acc, B)
                # bookkeeping
                losses.update(loss.item() / (2 * k**2), B)
                val_pos_acc.update(pos_acc, B)
                val_neg_acc.update(neg_acc, B)
                val_p_norm.update(pred_norm.item())
                val_g_norm.update(gt_norm.item())
            else:
                [score_, mask_] = model(input_seq)
                if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

                # [B, P, SQ, B, N, SQ]
                score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target_.view(B*NP*SQ, B2*NS*SQ)
                target_flattened = target_flattened.float().argmax(dim=1)

                loss = criterion(score_flattened, target_flattened)
                top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

                losses.update(loss.item(), B)
                accuracy.update(top1.item(), B)

                accuracy_list[0].update(top1.item(),  B)
                accuracy_list[1].update(top3.item(), B)
                accuracy_list[2].update(top5.item(), B)

    if args.hyperbolic and args.hyp_cone:
        print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
              'Acc: pos {2:.4f}; neg {3:.4f}; pnorm {4:.4f}; gnorm {5:.4f};\t'.format(
               epoch, args.epochs, pos_acc, neg_acc, pred_norm, gt_norm, loss=losses))
    else:
        print('[{0}/{1}] Loss {loss.local_avg:.4f}\t'
              'Acc: top1 {2:.4f}; top3 {3:.4f}; top5 {4:.4f} \t'.format(
               epoch, args.epochs, *[i.avg for i in accuracy_list], loss=losses))
    if args.hyperbolic and args.hyp_cone:
        return losses.local_avg, val_pos_acc.local_avg, val_neg_acc.local_avg, val_p_norm.local_avg, val_g_norm.local_avg
    else:
        return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    if args.dataset == 'k400':
        use_big_K400 = args.img_dim > 140
        dataset = Kinetics400_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              big=use_big_K400)
    elif args.dataset == 'k600':
        use_big_K600 = args.img_dim > 140
        dataset = Kinetics600_full_3d(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              big=use_big_K600)
    elif args.dataset == 'ucf101':
        dataset = UCF101_3d(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds)
    elif args.dataset == 'hollywood2':
        dataset = Hollywood2(mode=mode,
                             transform=transform,
                             seq_len=args.seq_len,
                             num_seq=args.num_seq,
                             downsample=args.ds)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset)

    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=100,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=sampler,
                                      shuffle=False,
                                      num_workers=32,
                                      pin_memory=True,
                                      drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_\
bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_\
train-{args.train_what}{2}'.format(
                    'r%s' % args.net[6::], \
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt=%s' % args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path

if __name__ == '__main__':
    main()
