import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU
from backbone.hyrnn_nets import MobiusGRU, MobiusLinear, MobiusDist2Hyperplane
import geoopt.manifolds.stereographic.math as gmath
import losses


class Model(nn.Module):
    '''DPC with RNN'''
    def __init__(self, args):

        self.args = args

        super(Model, self).__init__()
        torch.cuda.manual_seed(233)
        # print('Using DPC-RNN model')

        self.last_duration = int(math.ceil(args.seq_len / 4))
        self.last_size = int(math.ceil(args.img_dim / 32))

        self.target = self.sizes = None  # Only used if cross_gpu_score is True. Otherwise they are in trainer
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(args.network_feature, track_running_stats=False)
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU
        """
        When using a ConvGRU with a 1x1 convolution, it is equivalent to using a regular GRU by flattening the H and W 
        dimensions and adding those as extra samples in the batch (B' = BxHxW), and then going back to the original 
        shape.
        So we can use the hyperbolic GRU.
        """
        if args.hyperbolic:
            # Not used because the hyperbolic layer actually already works in Euclidean space as input
            # Layer to adapt from Euclidean to a value of Euclidean smaller than the one coming from the ResNet (at
            # least after initialization). Modifying the initializaton of the ResNet is hard because of the batchnorms
            # and the residual connections. And making the output after the network always small (by dividing by a fix
            # number is not ideal because the network cannot learn to calibrate).
            # self.adapt_layer = nn.Linear(self.param['feature_size'], self.param['feature_size'])
            # self._initialize_weights(self.adapt_layer, gain=0.01)

            self.hyperbolic_linear = MobiusLinear(self.param['feature_size'], self.param['feature_size'],
                                                  # This computes an exmap0 after the operation, where the linear
                                                  # operation operates in the Euclidean space.
                                                  hyperbolic_input=False,
                                                  hyperbolic_bias=True,
                                                  nonlin=None,  # For now
                                                  fp64_hyper=args.fp64_hyper
                                                  )
            if args.fp64_hyper:
                self.hyperbolic_linear = self.hyperbolic_linear.double()

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])

        if args.finetune or (args.early_action and not args.early_action_self):
            if args.hyperbolic:
                self.network_class = MobiusDist2Hyperplane(self.param['feature_size'], args.n_classes)
            else:
                self.network_class = nn.Linear(self.param['feature_size'], args.n_classes)
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

    def forward(self, block, labels=None):
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

        if self.args.hyperbolic:
            feature_reshape = feature.permute(0, 2, 3, 4, 1)
            mid_feature_shape = feature_reshape.shape
            feature_reshape = feature_reshape.reshape(-1, feature.shape[1])
            # feature_adapt = self.adapt_layer(feature_reshape)
            feature_dist = self.hyperbolic_linear(feature_reshape)  # performs exmap0
            # feature_dist = gmath.expmap0(feature_reshape, k=torch.tensor(-1.))
            feature_dist = feature_dist.reshape(mid_feature_shape).permute(0, 4, 1, 2, 3)

            if self.args.hyperbolic_version == 1:
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
        feature_dist = feature_dist[:, N-self.args.pred_step::, :].contiguous()
        feature_dist = feature_dist.permute(0,1,3,4,2).reshape(B*self.args.pred_step*self.last_size**2, self.param['feature_size'])  # .transpose(0,1)

        # ----------- STEP 2: compute predictions ------- #

        feature = self.relu(feature_g)  # [0, +inf)
        # [B,N,D,6,6], [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)

        hidden_all, hidden = self.agg(feature[:, 0:N-self.args.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

        if self.args.finetune or (self.args.early_action and not self.args.early_action_self):
            # pool
            pooled_hidden = hidden_all.mean(dim=[-2, -1]).view(-1, hidden_all.shape[2])  # just pool spatially
            # Predict label supervisedly
            if self.args.hyperbolic:
                feature_shape = pooled_hidden.shape
                pooled_hidden = pooled_hidden.view(-1, feature_shape[-1])
                if self.fp64_hyper:
                    pooled_hidden = pooled_hidden.double()
                pooled_hidden = self.hyperbolic_linear(pooled_hidden)
                pooled_hidden = pooled_hidden.view(feature_shape)
            pred_classes = self.network_class(pooled_hidden)
            pred = pred_classes
            size_pred = 1

        else:
            if self.args.early_action_self:
                # only one step but for all hidden_all, not just the last hidden
                pred = self.network_pred(hidden_all.view([-1] + list(hidden.shape[1:]))).view_as(hidden_all)

            else:
                pred = []
                for i in range(self.args.pred_step):
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

            if self.args.hyperbolic:
                pred = self.hyperbolic_linear(pred)

        sizes = self.last_size, self.args.pred_step, size_pred
        sizes = torch.tensor(sizes).to(pred.device).unsqueeze(0)

        if self.args.cross_gpu_score:  # Return all predictions to compute score for all gpus together
            return pred, feature_dist, sizes
        else:  # Compute scores individually for the data in this gpu
            sizes = sizes.float().mean(0).int()
            score = losses.compute_scores(self.args, pred, feature_dist, sizes, labels.shape[0])
            if self.target is None:
                self.target, self.sizes = losses.compute_mask(self.args, sizes, labels.shape[0])

            loss, *results = losses.compute_loss(self.args, score, pred, labels, self.target, self.sizes, labels.shape[0])
            return loss, results

    def _initialize_weights(self, module, gain=1.):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, gain)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
