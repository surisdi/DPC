import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as torch_dist
from tqdm import tqdm

import losses
import random
from utils.utils import save_checkpoint, AverageMeter


class Trainer:
    def __init__(self, args, model, optimizer, train_loader, val_loader, iteration, best_acc, writer_train, writer_val,
                 img_path, model_path, scheduler, partial=1.0):
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
        self.target = self.sizes = None
        self.partial = partial

    def train(self):
        # --- main loop --- #
        for epoch in range(self.args.start_epoch, self.args.epochs):
            accuracy_train_list = self.run_epoch(epoch, train=True)
            accuracy_val_list = self.run_epoch(epoch, train=False)

            if self.args.local_rank <= 0 and not self.args.debug:

                self.writers['train'].add_scalar('global/loss', accuracy_train_list['losses'], epoch)
                self.writers['val'].add_scalar('global/loss', accuracy_val_list['losses'], epoch)
                self.writers['train'].add_scalars('accuracy', accuracy_train_list, epoch)
                self.writers['val'].add_scalars('accuracy', accuracy_val_list, epoch)

                # save checkpoint
                is_best = accuracy_val_list['accuracy'] > self.best_acc
                self.best_acc = max(accuracy_val_list['accuracy'], self.best_acc)
                save_checkpoint({'epoch': epoch + 1,
                                 'net': self.args.network_feature,
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

        avg_meters = {k: AverageMeter() for k in ['losses', 'accuracy', 'hier_accuracy', 'top1', 'top3', 'top5', 'pos_acc', 'neg_acc',
                                                  'p_norm', 'g_norm', 'batch_time', 'data_time']}

        time_last = time.time()

        with tqdm(self.loaders['train' if train else 'val'], desc=f'Training epoch {epoch}' if train else
                  f'Evaluating {f"epoch {epoch}" if epoch else ""}', disable=self.args.local_rank > 0, \
                  total = int(len(self.loaders['train' if train else 'val']) * (self.partial if train else 1.0))) as t:
            for idx, (input_seq, labels) in enumerate(t):
                stop = int(len(self.loaders['train' if train else 'val']) * (self.partial if train else 1.0))
                if idx >= stop:
                    break
                # Measure data loading time
                avg_meters['data_time'].update(time.time() - time_last)

                input_seq = input_seq.to(self.args.device)
                labels = labels.to(self.args.device)

                # Get sequence predictions
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(train):
                        output_model = self.model(input_seq, labels)

                    if self.args.cross_gpu_score:
                        pred, feature_dist, sizes = output_model
                        sizes = sizes.float().mean(0).int()

                        if self.args.parallel == 'ddp':
                            tensors_to_gather = [pred, feature_dist, labels]
                            for i, v in enumerate(tensors_to_gather):
                                tensors_to_gather[i] = gather_tensor(v)
                            pred, feature_dist, labels = tensors_to_gather

                        score = losses.compute_scores(self.args, pred, feature_dist, sizes, labels.shape[0])
                        if self.target is None:
                            self.target, self.sizes = losses.compute_mask(self.args, sizes, labels.shape[0])

                        loss, *results = losses.compute_loss(self.args, score, pred, labels, self.target, self.sizes,
                                                   labels.shape[0])
                        del score
                    else:
                        loss, results = output_model
                    losses.bookkeeping(self.args, avg_meters, results)

                del input_seq

                if train:
                    # Backward pass
                    self.scaler.scale(loss.mean()).backward()
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

                if train and self.args.local_rank <= 0:
                    self.iteration += 1
                    if self.iteration % self.args.print_freq == 0 and self.writers['train'] and not self.args.debug:
                        num_outer_samples = (self.iteration + 1) * self.args.batch_size * \
                                            (1 if 'Parallel' in str(type(self.model)) else self.args.step_n_gpus)
                        self.writers['train'].add_scalars('train', {**postfix_kwargs}, num_outer_samples)

            if not train and self.args.local_rank <= 0:
                print(f'[{epoch}/{self.args.epochs}]' +
                      ''.join([f'{k}: {v.local_avg:.04f}, ' for k, v in avg_meters.items() if v.count > 0]))

            accuracy_list = {k: v.local_avg for k, v in avg_meters.items() if v.count > 0}

            return accuracy_list


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
