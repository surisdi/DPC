import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as torch_dist
from tqdm import tqdm

import losses
import random
from utils.utils import save_checkpoint, AverageMeter
import visualization

# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, args, model, optimizer, loaders, iteration, best_acc, writer_train, writer_val,
                 img_path, model_path, scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.iteration = iteration
        self.best_acc = best_acc
        self.writers = {'train': writer_train, 'val': writer_val}
        self.img_path = img_path
        self.model_path = model_path
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.target = self.sizes_mask = None

    def train(self):
        # --- main loop --- #
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.run_epoch(epoch, train=True)
            accuracy_val = self.run_epoch(epoch, train=False)

            if self.args.local_rank <= 0 and not self.args.debug:
                # save checkpoint
                is_best = accuracy_val > self.best_acc
                self.best_acc = max(accuracy_val, self.best_acc)
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

    def test(self):
        if self.args.test_info == 'compute_accuracy':
            accuracies_test = self.run_epoch(epoch=None, train=False)
            if self.args.local_rank <= 0:
                print('Accuracies test:')
                print(accuracies_test)
        elif self.args.test_info == 'generate_embeddings':
            visualization.generate_embeddings.main(self)
        elif self.args.test_info == 'viz_trajectories':
            visualization.viz_trajectories.main(self)
        elif self.args.test_info == 'viz_improvement':
            visualization.viz_improvement.main(self)
        elif self.args.test_info == 'viz_gradcam':
            visualization.viz_gradcam.main(self)
        elif self.args.test_info == 'extract_features':
            self.extract_features()
        else:
            print(f'Test {self.args.test_info} is not implemented')

    def run_epoch(self, epoch, train=True, return_all_acc=False):
        if self.args.device == "cuda":
            torch.cuda.synchronize()
        if train:
            self.model.train()
        else:
            self.model.eval()

        avg_meters = {k: AverageMeter() for k in ['losses', 'accuracy', 'hier_accuracy', 'top1', 'top3', 'top5',
                                                  'pos_acc', 'neg_acc', 'p_norm', 'g_norm', 'batch_time', 'data_time']}

        time_last = time.time()

        loader = self.loaders['train' if train else ('val' if epoch is not None else 'val')]
        desc = f'Training epoch {epoch}' if train else (f'Evaluating epoch {epoch}' if epoch is not None else 'Testing')
        stop_total = int(len(loader) * (self.args.partial))  # if train else 1.0))

        with tqdm(loader, desc=desc, disable=self.args.local_rank > 0, total=stop_total) as t:
            for idx, (input_seq, labels, *indices) in enumerate(t):
                if idx >= stop_total:
                    break
                # Measure data loading time
                avg_meters['data_time'].update(time.time() - time_last)
                input_seq = input_seq.to(self.args.device)
                labels = labels.to(self.args.device)

                # Get sequence predictions
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(train):
                        output_model, radius = self.model(input_seq, labels)

                    if self.args.cross_gpu_score:
                        pred, feature_dist, sizes_pred = output_model
                        sizes_pred = sizes_pred.float().mean(0).int()

                        if self.args.parallel == 'ddp':
                            tensors_to_gather = [pred, feature_dist, labels]
                            for i, v in enumerate(tensors_to_gather):
                                tensors_to_gather[i] = gather_tensor(v)
                            pred, feature_dist, labels = tensors_to_gather

                        if self.target is None:
                            self.target, self.sizes_mask = losses.compute_mask(self.args, sizes_pred, labels.shape[0])

                        loss, *results = losses.compute_loss(self.args, feature_dist, pred, labels, self.target, sizes_pred,
                                                             self.sizes_mask, labels.shape[0], indices, self)    # TODO remove indices
                    else:
                        loss, results = output_model
                    losses.bookkeeping(self.args, avg_meters, results)

                del input_seq

                if train:
                    # Backward pass
                    scaled_loss = self.scaler.scale(loss.mean())
                    scaled_loss.backward()
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
                    acuracy_list_train = {k: v for k, v in postfix_kwargs.items() if 'time' not in k}
                    self.iteration += 1
                    if self.iteration % self.args.print_freq == 0 and self.writers['train'] and not self.args.debug:
                        num_outer_samples = (self.iteration + 1) * self.args.batch_size * \
                                            (1 if 'Parallel' in str(type(self.model)) else self.args.step_n_gpus)
                        self.writers['train'].add_scalars('train', {**acuracy_list_train}, num_outer_samples)

            accuracy_list = {k: v.local_avg if train else v.avg for k, v in avg_meters.items()
                             if v.count > 0 and 'time' not in k}

            if not train and self.args.local_rank <= 0:
                print(f'[{epoch}/{self.args.epochs}]' +
                      ''.join([f'{k}: {", ".join([f"{v_:.04f}" for v_ in v.avg_expanded])}, '
                               for k, v in avg_meters.items() if v.count > 0]))
                if not self.args.debug:
                    self.writers['val'].add_scalar('global/loss', accuracy_list['losses'], epoch)
                    self.writers['val'].add_scalars('accuracy', accuracy_list, epoch)

            return accuracy_list if return_all_acc else accuracy_list['accuracy']

    def extract_features(self):

        split_extract = 'test'

        name_dataset = 'finegym' if 'FineGym' in self.args.path_dataset else'hollywood2'

        if self.args.device == "cuda":
            torch.cuda.synchronize()
        self.model.eval()

        loader = self.loaders[split_extract]
        stop_total = int(len(loader) * (self.args.partial))  # if train else 1.0))

        a_total = []
        b_total = []
        c_total = []
        percentages = []
        labels_total = []
        radius_total = []
        indices_total = []
        block_total = []

        with tqdm(loader, disable=self.args.local_rank > 0, total=stop_total) as t:
            for idx, (input_seq, labels, *indices) in enumerate(t):
                if idx >= stop_total:
                    break
                input_seq = input_seq.to(self.args.device)
                labels = labels.to(self.args.device)

                # Get sequence predictions
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(False):
                        output_model, radius = self.model(input_seq, labels, extract_features=True)

                a, b, c, labels, percent = output_model
                a_total.append(a)
                b_total.append(b)
                c_total.append(c)
                labels_total.append(labels)
                radius_total.append(radius)
                indices_total.append(indices[0])
                percentages.append(percent)

        a_total = torch.cat(a_total)
        b_total = torch.cat(b_total)
        if c_total[0] is not None:
            c_total = torch.cat(c_total)
        labels_total = torch.cat(labels_total)
        radius_total = torch.cat(radius_total)
        percentages = [torch.cat([perc[i] for perc in percentages]) for i in range(len(percentages[0]))]

        idxs_total = []
        for index in torch.cat(indices_total):
            if name_dataset == 'finegym':
                clip_idx = self.loaders[split_extract].dataset.idx2clipidx[index.item()]
                segments = self.loaders[split_extract].dataset.clips[clip_idx]
                start = (len(segments) - self.loaders[split_extract].dataset.num_seq) // 2
                action = list(segments.keys())[start + self.loaders[split_extract].dataset.num_seq - 1]
                idxs_total.append(clip_idx + '_' + action)
                block_total.append((segments, start))
            else:  # hollywood2
                vpath, vlen = self.loaders[split_extract].dataset.video_info.iloc[index.item()]
                items = self.loaders[split_extract].dataset.idx_sampler(vlen, vpath)
                idx_block, vpath = items

                idxs_total.append(vpath)
                block_total.append((idx_block, vlen))

        torch.save([a_total, b_total, c_total, labels_total, radius_total, idxs_total, block_total, percentages],
                   f'/proj/vondrick/didac/results/extracted_features_{name_dataset}_{split_extract}.pth')

    def get_base_model(self):
        if 'DataParallel' in str(type(self.model)):  # both the ones from apex and from torch.nn
            return self.model.module
        else:
            return self.model


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
