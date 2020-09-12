import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as torch_dist
from tqdm import tqdm

import losses
from utils.utils import save_checkpoint, AverageMeter


class Trainer:
    def __init__(self, args, model, optimizer, train_loader, val_loader, iteration, best_acc, writer_train, writer_val,
                 img_path, model_path, scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loaders = {'train': train_loader, 'val': val_loader}
        self.iteration = iteration
        self.best_acc = best_acc
        self.writers = {'train': writer_train, 'val': writer_val}
        self.img_path = img_path
        self.model_path = model_path
        self.scheduler = scheduler
        self.scaler = GradScaler()

    def train(self):
        # --- main loop --- #
        for epoch in range(self.args.start_epoch, self.args.epochs):
            accuracy_train_list = self.run_epoch(epoch, train=True)
            accuracy_val_list = self.run_epoch(epoch, train=False)

            self.writers['train'].add_scalar('global/loss', accuracy_train_list['losses'], epoch)
            self.writers['val'].add_scalar('global/loss', accuracy_val_list['losses'], epoch)
            self.writers['train'].add_scalar('accuracy', accuracy_train_list, epoch)
            self.writers['val'].add_scalars('accuracy', accuracy_val_list, epoch)

            # save checkpoint
            is_best = accuracy_val_list['accuracy'] > self.best_acc
            self.best_acc = max(accuracy_val_list['accuracy'], self.best_acc)

            if self.args.local_rank <= 0 and not self.args.debug:
                save_checkpoint({'epoch': epoch + 1,
                                 'net': self.args.feature_network,
                                 'state_dict': (self.model.module if hasattr(self.model, 'module') else
                                                self.model).state_dict(),
                                 'best_acc': self.best_acc,
                                 'optimizer': self.optimizer.state_dict(),
                                 'iteration': self.iteration,
                                 'scheduler': self.scheduler.state_dict()},
                                is_best, filename=os.path.join(self.model_path, f'epoch{epoch+1}.pth.tar'),
                                keep_all=False)

    def run_epoch(self, epoch, train=True):
        if self.args.device == "cuda":
            torch.cuda.synchronize()
        if train:
            self.model.train()
        else:
            self.model.eval()

        avg_meters = {k: AverageMeter() for k in ['losses, accuracy', 'top1', 'top3' 'top5', 'pos_acc', 'neg_acc',
                                                  'p_norm', 'g_norm', 'batch_time', 'data_time']}

        time_last = time.time()

        with tqdm(self.loaders['train' if train else 'val'], desc=f'Training epoch {epoch}' if train else
                  f'Evaluating {f"epoch {epoch}" if epoch else ""}', disable=self.args.local_rank > 0) as t:
            for idx, (input_seq, labels) in enumerate(t):
                # Measure data loading time
                avg_meters['data_time'].update(time.time() - time_last)

                input_seq = input_seq.to(self.args.device)
                B = input_seq.size(0)

                # Get sequence predictions
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(train):
                        score, mask, pred_norm, gt_norm = self.model(input_seq)

                    if self.args.parallel == 'ddp':
                        tensors_to_gather = [score, mask, pred_norm, gt_norm]
                        for i, v in enumerate(tensors_to_gather):
                            tensors_to_gather[i] = gather_tensor(v)
                        score, mask, pred_norm, gt_norm = tensors_to_gather

                    if idx == 0:
                        target, sizes = process_output(mask)
                    loss = losses.compute_loss(self.args, score, pred_norm, gt_norm, labels, target, sizes,
                                               avg_meters, B)

                del score, input_seq

                if train:
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                del loss

                avg_meters['batch_time'].update(time.time() - time_last)
                time_last = time.time()

                # ------------- Show information ------------ #
                postfix_kwargs = {k: v.val for k, v in avg_meters.items() if v.count > 0}
                t.set_postfix(**postfix_kwargs)

                if train:
                    self.iteration += 1
                    if self.iteration % self.args.print_freq == 0 and self.writers['train'] and not self.args.debug:
                        num_outer_samples = (self.iteration + 1) * self.args.batch_size * \
                                            (1 if 'Parallel' in str(type(self.model)) else self.args.step_n_gpus)
                        self.writers['train'].add_scalars('train', {**postfix_kwargs}, num_outer_samples)

            if not train and self.args.local_rank <= 0:
                print(f'[{epoch}/{self.args.epochs}] Loss {loss.local_avg:.4f}\t' +
                      ''.join([f'{k}: {v.local_avg}' for k, v in avg_meters.items() if v.count > 0]))

            accuracy_list = {k: v.local_avg for k, v in avg_meters.items() if v.count > 0}

            return accuracy_list


def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)


def gather_tensor(v):
    if v is None:
        return None

    # list where each element is [N x H_DIM]
    gather_dest = [torch.empty_like(v) * i for i in range(torch_dist.get_world_size())]
    torch_dist.all_gather(gather_dest, v.contiguous())  # as far as i recall, this loses gradient information completely

    gather_dest[torch_dist.get_rank()] = v  # restore tensor with gradient information
    gather_dest = torch.cat(gather_dest)

    # gather_dest is now a tensor of [(N*N_GPUS) x H_DIM], as if you ran everything on one GPU, except only N samples
    # corresponding to GPU i inputs will have gradient information
    return gather_dest
