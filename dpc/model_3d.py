import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
sys.path.append('../util')
from hyp_cone import HypConeDist
from select_backbone import select_resnet
from hyptorch_math import dist_matrix
from convrnn import ConvGRU
from hyrnn_nets import MobiusGRU, MobiusLinear, MobiusDist2Hyperplane
import geoopt.manifolds.stereographic.math as gmath
import geoopt


class DPC_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network='resnet50', loss='dot',
                 hyperbolic_version=1, distance='regular', margin=0.1, early_action=False,
                 early_action_self=False, nclasses=0):
        super(DPC_RNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        self.margin = margin
        self.nclasses = nclasses
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU
        """
        When using a ConvGRU with a 1x1 convolution, it is equivalent to using a regular GRU by flattening the H and W 
        dimensions and adding those as extra samples in the batch (B' = BxHxW), and then going back to the original 
        shape.
        So we can use the hyperbolic GRU.
        """
        self.hyperbolic = hyperbolic
        self.hyperbolic_version = hyperbolic_version
        self.distance = distance
        self.hyp_cone = hyp_cone
        self.margin=margin
        self.early_action = early_action
        self.early_action_self = early_action_self
        if hyperbolic:
            self.hyperbolic_linear = MobiusLinear(self.param['feature_size'], self.param['feature_size'],
                                                  # This computes an exmap0 after the operation, where the linear
                                                  # operation operates in the Euclidean space.
                                                  hyperbolic_input=False,
                                                  hyperbolic_bias=True,
                                                  nonlin=None,  # For now
                                                  ).double()

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])

        if self.early_action and not self.early_action_self:
            if hyperbolic:
                self.network_class = MobiusDist2Hyperplane(self.param['feature_size'], self.nclasses)
            else:
                self.network_class = nn.Linear(self.param['feature_size'], self.nclasses)
        else:
            self.network_pred = nn.Sequential(
                                    nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                    )
            self._initialize_weights(self.network_pred)

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)


    def forward(self, block):
        a = time.time()
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape

        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        if self.hyperbolic and self.hyperbolic_version == 2:
            feature_shape = feature.shape
            feature_hyp = feature.permute(0,2,3,4,1).contiguous() / 10
            feature_hyp_shape = feature_hyp.shape
            feature_hyp = feature_hyp.view(-1, feature.shape[1]).double()
            feature_hyp = self.hyperbolic_linear(feature_hyp)
            feature_hyp = feature_hyp.view(feature_hyp_shape)
            feature_hyp = feature_hyp.permute(0,4,1,2,3)
            assert feature_hyp.shape == feature_shape

            feature_inf_all = feature_hyp.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)

            # project back to euclidean
            feature = gmath.logmap0(feature_hyp, k=torch.tensor(-1.), dim=1).float()

        else:
            feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)

        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()

        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)

        del feature_inf_all

        # ### aggregate, predict future ###
        # if self.hyperbolic:
        #     feature_shape = feature.shape
        #     feature = feature.view(-1, feature_shape[-1])
        #     _, hidden = self.huperbolic_agg(feature[:, 0:N - self.pred_step, :].contiguous())
        #     hidden = hidden.view(feature_shape)
        # else:
        hidden_all, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

        if self.early_action and not self.early_action_self:
            # pool
            pooled_hidden = hidden_all.mean(dim=[-2, -1]).view(-1, hidden_all.shape[2])  # just pool spatially
            # Predict label supervisedly
            if self.hyperbolic:
                feature_shape = pooled_hidden.shape
                pooled_hidden = pooled_hidden.view(-1, feature_shape[-1]).double() / 10
                pooled_hidden = self.hyperbolic_linear(pooled_hidden)
                pooled_hidden = pooled_hidden.view(feature_shape)
            pred_classes = self.network_class(pooled_hidden)

            return pred_classes, None

        if self.early_action_self:
            # only one step but for all hidden_all, not just the last hidden
            pred = self.network_pred(hidden_all.view([-1] + list(hidden.shape[1:]))).view_as(hidden_all)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(B * hidden_all.shape[1] * self.last_size ** 2,
                                                                 self.param['feature_size'])

        else:
            pred = []
            for i in range(self.pred_step):
                # sequentially pred future

                # if self.loss == 'hyp_poincare':
                #     p_tmp = self.hyperbolic_network_pred(hidden)
                #     pred.append(p_tmp)
                #     p_tmp_shape = p_tmp.shape
                #     p_tmp = p_tmp.view(-1, p_tmp_shape[-1])
                #     _, hidden = self.hyperbolic_agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
                #     hidden = hidden.view(p_tmp_shape)
                # else:
                p_tmp = self.network_pred(hidden)
                pred.append(p_tmp)
                _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
                hidden = hidden[:,-1,:]
            pred = torch.stack(pred, 1) # B, pred_step, xxx
            pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(B * self.pred_step * self.last_size ** 2,
                                                                 self.param['feature_size'])

            del hidden


        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size'])  #.transpose(0,1)

        b = time.time()
        if self.hyperbolic:
            # TODO clean and make less "if-else", and more general.
            if self.hyperbolic_version == 1:
                feature_shape = feature_inf.shape
                feature_inf_hyp = feature_inf.view(-1, feature_shape[-1]).double()
                feature_inf_hyp = self.hyperbolic_linear(feature_inf_hyp)
                # feature_inf_hyp = gmath.expmap0(feature_inf_hyp, k=torch.tensor(-1.))
                feature_inf_hyp = feature_inf_hyp.view(feature_shape)

            else:  # hyperbolic2
                feature_inf_hyp = feature_inf  # was already in hyperbolic space

            pred_shape = pred.shape
            pred_hyp = pred.view(-1, pred_shape[-1]).double()
            pred_hyp = self.hyperbolic_linear(pred_hyp)
            # pred_hyp = gmath.expmap0(pred_hyp, k=torch.tensor(-1.))
            pred_hyp = pred_hyp.view(pred_shape)

            pred_norm, gt_norm = None, None
            if self.hyp_cone:
                shape_expand = (pred_hyp.shape[0], pred_hyp.shape[0], pred_hyp.shape[1])
                dist_fn = HypConeDist(K=0.1)
                pred_flatten = pred_hyp.unsqueeze(1).expand(shape_expand).contiguous().view(-1, shape_expand[-1])
                gt_flatten = feature_inf_hyp.unsqueeze(0).expand(shape_expand).contiguous().view(-1, shape_expand[-1])
                score = dist_fn(pred_flatten, gt_flatten)

                # loss function (equation 32 of https://arxiv.org/abs/1804.01882)
                score = score.reshape(B * self.pred_step * self.last_size ** 2,
                                      B * self.pred_step * self.last_size ** 2)

                # TODO put as option. Ruoshi version
                # pred_norm = torch.mean(torch.norm(pred_flatten, p=2, dim=-1))
                # gt_norm = torch.mean(torch.norm(gt_flatten, p=2, dim=-1))
                # score[score < 0] = 0
                # pos = score.diagonal(dim1=-2, dim2=-1)
                # score = self.margin - score
                # score[score < 0] = 0
                # k = score.size(0)
                # # positive score is multiplied by number of negative samples
                # weight = 1
                # score.as_strided([k], [k + 1]).copy_(pos * (k - 1) * weight);

            else:
                # distance can also be computed with geoopt.manifolds.PoincareBall(c=1) -> .dist
                # Maybe more accurate (it is more specific for poincare)
                # But this is much faster... TODO implement batch dist_matrix on geoopt library
                # score = dist_matrix(pred_hyp, feature_inf_hyp)

                manif = geoopt.manifolds.PoincareBall(c=1)
                shape_expand = (pred_hyp.shape[0], feature_inf_hyp.shape[0], pred_hyp.shape[1])
                score = manif.dist(pred_hyp.unsqueeze(1).expand(shape_expand).contiguous().view(-1, shape_expand[-1]),
                                   feature_inf_hyp.unsqueeze(0).expand(shape_expand).contiguous().view(-1, shape_expand[-1])
                                   ).view(shape_expand[:2])
                if self.distance == 'squared':
                    score = score.pow(2)
                elif self.distance == 'cosh':
                    score = torch.cosh(score).pow(2)
                score = - score.float()
                pred_temp_size = self.num_seq -1 if self.early_action_self else self.pred_step
                score = score.view(B, pred_temp_size, self.last_size**2, B, N, self.last_size**2)

        else: # euclidean dot product
            score = torch.matmul(pred, feature_inf.transpose(0,1))
            score = score.view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)

        c = time.time()

        del feature_inf, pred

        if self.mask is None:  # only compute mask once
            self.compute_mask(B, N)

        d = time.time()

        # print(b-a, c-b, d-c)

        return score, self.mask, pred_norm, gt_norm

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

    def compute_mask(self, B, N):
        # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
        if self.early_action_self:
            # Here NO temporal neg! All steps try to predict the last one
            mask = torch.zeros((B, self.num_seq - 1, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
                               requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.num_seq - 1,
                                                                   B * self.last_size ** 2, N)
            for j in range(B * self.last_size ** 2):
                tmp[j, torch.arange(self.num_seq - 1), j, torch.arange(N - self.pred_step, N)] = 1  # pos
            mask = tmp.view(B, self.last_size ** 2, self.num_seq - 1, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5,
                                                                                                         4)
            self.mask = mask
        else:
            mask = torch.zeros((B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
                               requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size ** 2), k, :,
                torch.arange(self.last_size ** 2)] = -1  # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.pred_step,
                                                                   B * self.last_size ** 2, N)
            for j in range(B * self.last_size ** 2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos
            mask = tmp.view(B, self.last_size ** 2, self.pred_step, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)
            self.mask = mask

