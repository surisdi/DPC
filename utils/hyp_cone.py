import torch

class HypConeDist():
    def __init__(self, K=0.1):
        self.K = K
    def __call__(self, x, y):
        '''
        scale up embedding if it's smaller than the threshold radius K
        
        Note: this step potentially contains a lot of in-place operation,
        which is not legal for torch.autograd. Need to make clone of
        the variable every step of the way
        '''
        x_norm = torch.norm(x.clone(), p=2, dim=-1)
        x_small = x[x_norm < (self.K + 1e-7)].clone().transpose(dim0=-1, dim1=-2)
        scale_factor = ((self.K + 1e-7) / x_norm[x_norm < (self.K + 1e-7)].clone())
        x_small_clone = (x_small * scale_factor).transpose(dim0=-1, dim1=-2)
        x[x_norm < (self.K + 1e-7)] = x_small_clone.clone()
        return self.Xi(x, y) - self.Phi(x, K = self.K)
    def Xi(self, x, y):
        x_norm = torch.norm(x, p=2, dim=-1).double()
        y_norm = torch.norm(y, p=2, dim=-1).double()
        xy_dot = (x * y).sum(dim=-1).double()
        w = x_norm * torch.norm(x - y, p=2, dim=-1).double()
        up = xy_dot * (1 + torch.pow(x_norm, 2)) - torch.pow(x_norm, 2) * (1 + torch.pow(y_norm, 2))
        bot = w * torch.sqrt((1 + torch.pow(x_norm, 2) * torch.pow(y_norm, 2) - 2 * xy_dot))
        eps = 1e-6
        division = up / bot
        xi = torch.acos(up / bot - eps*division.sign())
        return xi
    def Phi(self, x, K=0.1):
        x_norm = torch.norm(x, p=2, dim=-1)
        return torch.asin(K * (1 - torch.pow(x_norm, 2)) / x_norm)