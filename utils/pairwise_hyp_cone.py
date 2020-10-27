import torch
import numpy as np

class PairwiseHypConeDist():
    def __init__(self, K=0.1, fp64_hyper=True):
        self.K = K
        self.fp64_hyper = fp64_hyper
    def __call__(self, x, y):
        '''
        scale up embedding if it's smaller than the threshold radius K
        
        Note: this step potentially contains a lot of in-place operation,
        which is not legal for torch.autograd. Need to make clone of
        the variable every step of the way
        '''
        N_pred, D = x.shape
        N_gt, D = y.shape
        
        # scaling up x when it's too small
        x_norm = torch.norm(x, p=2, dim=-1)
        x_small = x.transpose(dim0=-1, dim1=-2)
        scale_factor = ((0.1 + 1e-7) / x_norm)
        x_small = (x_small * scale_factor).clone()
        x = torch.where(x_norm < (0.1 + 1e-7), x_small, x.transpose(dim0=-1, dim1=-2)).transpose(dim0=-1, dim1=-2)
        
        # neccessary components
        x_square = self.square_norm(x).unsqueeze(dim=1).expand(N_pred, N_gt)
        y_square = self.square_norm(y).unsqueeze(dim=0).expand(N_pred, N_gt)
        x_norm = torch.sqrt(x_square)
        xy_square = self.pairwise_distances(x, y)
        xy_norm = torch.sqrt(xy_square)
        xy_prod = self.pairwise_mul(x, y)
        
        # Xi
        num = xy_prod * (1 + x_square) - x_square * (1 + y_square)
        denom = x_norm * xy_norm * torch.sqrt(1 + x_square * y_square - 2.0 * xy_prod)
        Xi = torch.acos(num / denom)
        
        # Phi
        Phi = torch.asin(self.K * (1 - x_square) / x_norm)
        
        return Xi - Phi
    
    def square_norm(self, x):
        """
        Helper function returning square of the euclidean norm.
        Also here we clamp it since it really likes to die to zero.
        """
        norm = torch.norm(x,dim=-1,p=2)**2
        return torch.clamp(norm, min=0.0)

    def pairwise_mul(self, x, y):
        """
        Helper function returning pairwise vector product.
        Also here we clamp it since it really likes to die to zero.
        """
        y_t = torch.transpose(y, 0, 1)
        prod = torch.mm(x, y_t)
        return prod

    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)