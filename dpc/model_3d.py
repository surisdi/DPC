import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from select_backbone import select_resnet
from hyptorch_math import dist_matrix
from convrnn import ConvGRU
from hyrnn_nets import MobiusGRU, MobiusLinear
import geoopt.manifolds.stereographic.math as gmath
import geoopt


class DPC_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network='resnet50', hyperbolic='euclidean'):
        super(DPC_RNN, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
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
        if 'hyperbolic' in hyperbolic:
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

        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)


    def forward(self, block):
        a = time.time()
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        if self.hyperbolic == 'hyperbolic2':
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
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future

            # if self.hyperbolic:
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
        del hidden


        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT. 
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size'])  #.transpose(0,1)

        b = time.time()
        if 'hyperbolic' in self.hyperbolic:

            if self.hyperbolic == 'hyperbolic1' or self.hyperbolic == 'hyperbolic3' or self.hyperbolic == 'hyperbolic4':
                feature_shape = feature_inf.shape
                feature_inf_hyp = feature_inf.view(-1, feature_shape[-1]).double()/10
                feature_inf_hyp = self.hyperbolic_linear(feature_inf_hyp)
                # feature_inf_hyp = gmath.expmap0(feature_inf_hyp, k=torch.tensor(-1.))
                feature_inf_hyp = feature_inf_hyp.view(feature_shape)

            else:  # hyperbolic2
                feature_inf_hyp = feature_inf  # was already in hyperbolic space

            pred_shape = pred.shape
            pred_hyp = pred.view(-1, pred_shape[-1]).double()/10
            pred_hyp = self.hyperbolic_linear(pred_hyp)
            # pred_hyp = gmath.expmap0(pred_hyp, k=torch.tensor(-1.))
            pred_hyp = pred_hyp.view(pred_shape)

            # distance can also be computed with geoopt.manifolds.PoincareBall(c=1) -> .dist
            # Maybe more accurate (it is more specific for poincare)
            # But this is much faster... TODO implement batch dist_matrix on geoopt library
            # score = dist_matrix(pred_hyp, feature_inf_hyp)
            #
            manif = geoopt.manifolds.PoincareBall(c=1)
            shape_expand = (pred_hyp.shape[0], pred_hyp.shape[0], pred_hyp.shape[1])
            score = manif.dist(pred_hyp.unsqueeze(1).expand(shape_expand).contiguous().view(-1, shape_expand[-1]),
                               feature_inf_hyp.unsqueeze(0).expand(shape_expand).contiguous().view(-1, shape_expand[-1])
                               ).view(shape_expand[:2])
            if self.hyperbolic == 'hyperbolic3':
                score = score.pow(2)
            elif self.hyperbolic == 'hyperbolic4':
                score = torch.cosh(score).pow(2)
            score = - score.float()

        else:
            score = torch.matmul(pred, feature_inf.transpose(0,1))
        score = score.view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)

        c = time.time()

        del feature_inf, pred

        if self.mask is None: # only compute mask once
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
            for j in range(B*self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
            mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
            self.mask = mask

        d = time.time()

        # print(b-a, c-b, d-c)


        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None

