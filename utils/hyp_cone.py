import torch

class HypConeDist():
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
        x_norm = torch.norm(x, p=2, dim=-1)
        x_small = x.transpose(dim0=-1, dim1=-2)
        scale_factor = ((0.1 + 1e-7) / x_norm)
        x_small = (x_small * scale_factor).clone()
        x = torch.where(x_norm < (0.1 + 1e-7), x_small, x.transpose(dim0=-1, dim1=-2)).transpose(dim0=-1, dim1=-2)
        return self.Xi(x, y) - self.Phi(x, K = self.K)
    def Xi(self, x, y):
        x_norm = torch.norm(x, p=2, dim=-1)
        y_norm = torch.norm(y, p=2, dim=-1)
        difference = torch.norm(x - y, p=2, dim=-1)
        xy_dot = (x * y).sum(dim=-1)
        if self.fp64_hyper:
            x_norm = x_norm.double()
            y_norm = y_norm.double()
            difference = difference.double()
            xy_dot = xy_dot.double()

        w = x_norm * difference
        up = xy_dot * (1 + torch.pow(x_norm, 2)) - torch.pow(x_norm, 2) * (1 + torch.pow(y_norm, 2))
        bot = w * torch.sqrt((1 + torch.pow(x_norm, 2) * torch.pow(y_norm, 2) - 2 * xy_dot))
        eps = 1e-6
        division = up / bot
        xi = torch.acos(up / bot - eps*division.sign())
        return xi
    def Phi(self, x, K=0.1):
        x_norm = torch.norm(x, p=2, dim=-1)
        return torch.asin(K * (1 - torch.pow(x_norm, 2)) / x_norm)