import torch
import torch.nn as nn

@torch.no_grad()
def smooth_one_hot(target: torch.Tensor, num_classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    true_dist = target.new_zeros(size=(len(target), num_classes)).float()
    true_dist.fill_(smoothing / (num_classes - 1))
    true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return true_dist

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0,reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction=reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bs = float(pred.size(0))
        pred = pred.log_softmax(dim=1)
        if len(target.shape) == 2:
            true_dist = target
        else:
            true_dist = smooth_one_hot(target, self.num_classes, self.smoothing)
        loss = (-pred * true_dist).sum(dim=1)
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='none':
            return loss
        

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, indices=None, lam=1.,lam_b=None, smoothing=0.0, device='cuda', target_b=None):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    if target_b is None:
        y2=y1
    else:
        if indices is None: # not mix
            y2 = one_hot(target_b, num_classes, on_value=on_value, off_value=off_value, device=device)
        else: # mix
            y2 = one_hot(target[indices], num_classes, on_value=on_value, off_value=off_value, device=device)
    if lam_b is None:
        lam_b=1.-lam
    if isinstance(lam, torch.Tensor):
        if len(lam.shape)==1:
            lam.unsqueeze_(1)
            lam_b.unsqueeze_(1)
    return y1 * lam + y2 * lam_b