import torch

## min-max - norm ##

def minmax_norm(source,eps=1e-12):
    norm = source.view(source.size(0), -1)
    min_norm=norm.min(1, keepdim=True)[0]
    max_norm=norm.max(1, keepdim=True)[0]
    norm = (norm-min_norm+eps)/(max_norm-min_norm+eps).detach()

    norm = norm.view_as(source)
    return norm

## transform norm-denorm
class DeNormalize:
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    
    def __call__(self,tensor):
        dtype=tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

class Normalize:
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
    
    def __call__(self,tensor):
        dtype=tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.sub_(mean).div_(std)
        return tensor