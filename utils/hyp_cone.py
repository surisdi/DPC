import torch

class HypConeDist():
    def __init__(self, K=0.1):
        self.K = K
    def __call__(self, x, y):
        return self.Xi(x, y) - self.Phi(x, K = self.K)
    def Xi(self, x, y):
        x_norm = torch.norm(x, p=2, dim=-1)
        y_norm = torch.norm(y, p=2, dim=-1)
        xy_dot = (x * y).sum(dim=-1)
        w = x_norm * torch.norm(x - y, p=2, dim=-1)
        up = xy_dot * (1 + torch.pow(x_norm, 2)) - torch.pow(x_norm, 2) * (1 + torch.pow(y_norm, 2))
        bot = w * torch.sqrt((1 + torch.pow(x_norm, 2) * torch.pow(y_norm, 2) - 2 * xy_dot))
        return torch.acos(up / bot)
    def Phi(self, x, K=0.1):
        x_norm = torch.norm(x, p=2, dim=-1)
        return torch.asin(K * (1 - torch.pow(x_norm, 2)) / x_norm)