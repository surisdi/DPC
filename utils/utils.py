import torch
import numpy as np
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import deque
from torchvision import transforms


def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename),
                                   'epoch%s.pth.tar' % str(state['epoch']-gap))
    if not keep_all:
        try: os.remove(last_epoch_path)
        except: pass
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        for i in past_best:
            try: os.remove(i)
            except: pass
        path_best = os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch']))
        torch.save(state, path_best)
        print(f'Updating best model: {path_best}')


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def calc_accuracy(output, target):
    '''output: (B, N); target: (B)'''
    target = target.squeeze()
    _, pred = torch.max(output, 1)
    return torch.mean((pred == target).float())


def calc_accuracy_binary(output, target):
    '''output, target: (B, N), output is logits, before sigmoid '''
    pred = output > 0
    acc = torch.mean((pred == target.byte()).float())
    del pred, output, target
    return acc


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean)==len(std)==3
    inv_mean = [-mean[i]/std[i] for i in range(3)]
    inv_std = [1/i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table
        self.avg_expanded = None

    def update(self, val, n=1, history=0, step=5):
        is_array = False
        if type(val) == torch.Tensor:
            if len(val.shape) > 0 and val.shape[0] > 1:
                is_array = True
                val = val.view(-1).cpu().data.detach().numpy()
            else:
                val = val.mean().item()
        elif type(val) == np.ndarray:
            if len(val.shape) > 0 and val.shape[0] > 1:
                is_array = True
                val = val.reshape(-1)
        elif type(val) == float:
            pass
        else:
            raise TypeError(f'{type(val)} type not supported in AverageMeter')

        if type(n) == torch.Tensor:
            n = n.float().mean().item()
        self.val = np.mean(val)
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        if is_array:
            self.avg_expanded = self.sum / self.count
            self.avg = self.avg_expanded.mean()
        else:
            self.avg = self.sum / self.count
            self.avg_expanded = np.array([self.avg])
        if history:
            self.history.append(val.mean())
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class AccuracyTable(object):
    '''compute accuracy for each class'''
    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count':0,'correct':0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


def neq_load_customized(args, model, pretrained_dict,
                        parts=['backbone', 'agg', 'network_pred', 'hyperbolic_linear', 'network-class'],
                        size_diff=False):
    '''
    load pre-trained model in a not-equal way, when new model has been partially modified
    size_diff: some parameters may have the same name but different size. Cannot load these, but do not throw error, and
    load all the rest
    '''
    model_dict = model.state_dict()
    tmp = {}
    print_r(args, '\n=======Check Weights Loading======')
    print_r(args, ('loading the following parts:', ', '.join(parts)))
    if parts == 'all':
        if size_diff:
            for k, v in pretrained_dict.items():
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    tmp[k] = v
                else:
                    print_r(args, f'{k} not loaded')
        else:
            tmp = pretrained_dict
    else:
        for part in parts:
            print_r(args, ('loading:', part))
            print_r(args, '\n=======Check Weights Loading======')
            print_r(args, 'Weights not used from pretrained file:')
            for k, v in pretrained_dict.items():
                if part in k:
                    if k in model_dict:
                        if not (size_diff and model.state_dict()[k].shape != v.shape):
                            tmp[k] = v
                    else:
                        print_r(args, k)
            print_r(args, '---------------------------')
            print_r(args, 'Weights not loaded into new model:')
            for k, v in model_dict.items():
                if part in k:
                    if k not in pretrained_dict:
                        print_r(args, k)
            print_r(args, '===================================\n')

    del pretrained_dict
    if 'time_index.weight' in tmp and \
            'time_index' in [a[0].split('.')[0] for a in list(model.named_parameters())] and \
            model.time_index.weight.shape[0] < tmp['time_index.weight'].shape[0]:
        tmp['time_index.weight'].data = tmp['time_index.weight'][:model.time_index.weight.shape[0]].data
    model.load_state_dict(tmp, strict=False)
    return model


def print_r(args, text, print_no_verbose=False):
    """ Print only when the local rank is <=0 (only once)"""
    if args.local_rank <= 0 and (args.verbose or print_no_verbose):
        if type(text) == tuple:
            print(*text)
        else:
            print(text)
            
            
class ConfusionMeter(object):
    '''compute and show confusion matrix'''
    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class))
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p,t in zip(pred.flat, tar.flat):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
            cmap=plt.cm.jet,
            interpolation=None,
            extent=(0.5, np.shape(self.mat)[0]+0.5, np.shape(self.mat)[1]+0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y+1, x+1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i+1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i+1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))




