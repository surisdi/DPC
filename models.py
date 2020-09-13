import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU
from backbone.hyrnn_nets import MobiusGRU, MobiusLinear, MobiusDist2Hyperplane
import geoopt.manifolds.stereographic.math as gmath


class Model(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network_feature='resnet50', hyperbolic=False,
                 hyperbolic_version=1, hyp_cone=False, distance='regular', margin=0.1, early_action=False,
                 early_action_self=False, nclasses=0, downstream=False):
        super(Model, self).__init__()
        torch.cuda.manual_seed(233)
        # print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        self.margin = margin
        self.nclasses = nclasses
        self.downstream = downstream
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network_feature, track_running_stats=False)
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
        self.margin = margin
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

        if downstream or (self.early_action and not self.early_action_self):
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
        (B, N, C, SL, H, W) = block.shape

        # ----------- STEP 1: compute features ------- #
        # features_dist are the features used to compute the distance
        # features_g are the features used to input to the prediction network

        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        if self.hyperbolic:
            feature_reshape = feature.permute(0, 2, 3, 4, 1)
            mid_feature_shape = feature_reshape.shape
            feature_reshape = feature_reshape.reshape(-1, feature.shape[1])
            feature_dist = self.hyperbolic_linear(feature_reshape)
            # feature_dist = gmath.expmap0(feature_reshape, k=torch.tensor(-1.))
            feature_dist = feature_dist.reshape(mid_feature_shape).permute(0, 4, 1, 2, 3)

            if self.hyperbolic_version == 1:
                # Do not modify Euclidean feature for g, but compute hyperbolic version for the distance later
                feature_g = feature

            else:  # hyperbolic version 2
                # Move to hyperbolic with linear layer, then create feature_g from there (back to Euclidean)
                # project back to euclidean
                feature_g = gmath.logmap0(feature_dist, k=torch.tensor(-1.), dim=1).float()
        else:
            feature_dist = feature_g = feature

        # before ReLU, (-inf, +inf)
        feature_dist = feature_dist.view(B, N, self.param['feature_size'], self.last_size, self.last_size)
        feature_dist = feature_dist[:, N-self.pred_step::, :].contiguous()
        feature_dist = feature_dist.permute(0,1,3,4,2).reshape(B*self.pred_step*self.last_size**2, self.param['feature_size'])  # .transpose(0,1)

        # ----------- STEP 2: compute predictions ------- #

        feature = self.relu(feature_g)  # [0, +inf)
        # [B,N,D,6,6], [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)

        hidden_all, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

        if self.downstream or (self.early_action and not self.early_action_self):
            # pool
            pooled_hidden = hidden_all.mean(dim=[-2, -1]).view(-1, hidden_all.shape[2])  # just pool spatially
            # Predict label supervisedly
            if self.hyperbolic:
                feature_shape = pooled_hidden.shape
                pooled_hidden = pooled_hidden.view(-1, feature_shape[-1]).double()
                pooled_hidden = self.hyperbolic_linear(pooled_hidden)
                pooled_hidden = pooled_hidden.view(feature_shape)
            pred_classes = self.network_class(pooled_hidden)
            pred = pred_classes
            size_pred = 1

        else:
            if self.early_action_self:
                # only one step but for all hidden_all, not just the last hidden
                pred = self.network_pred(hidden_all.view([-1] + list(hidden.shape[1:]))).view_as(hidden_all)

            else:
                pred = []
                for i in range(self.pred_step):
                    # sequentially pred future
                    p_tmp = self.network_pred(hidden)
                    pred.append(p_tmp)
                    _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
                    hidden = hidden[:,-1,:]
                pred = torch.stack(pred, 1)  # B, pred_step, xxx

                del hidden

            size_pred = pred.shape[1]
            pred = pred.permute(0, 1, 3, 4, 2)
            pred = pred.reshape(-1, pred.shape[-1])

            if self.hyperbolic:
                pred = self.hyperbolic_linear(pred)

        sizes = self.last_size, self.pred_step, size_pred

        return pred, feature_dist, sizes

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
