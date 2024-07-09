import numpy as np
import torch 
from Mixup.utils import rand_bbox, saliency_bbox

def input_mixup(inputs, targets, indices=None , alpha=0.0):
    if alpha ==0.0:
        mixup_lam=1.0
    else: mixup_lam = np.random.beta(alpha, alpha)
    if indices is None:
        indices = np.random.permutation(inputs.shape[0])

    targets_b = targets[indices]
    feature_b = inputs[indices]
    mixup_feature = mixup_lam*inputs + (1-mixup_lam)*feature_b
    mixup_lam = torch.ones(inputs.shape[0], device=inputs.device)*mixup_lam

    return mixup_feature, targets, targets_b, mixup_lam,1-mixup_lam


def cutmix(inputs, targets, indices=None, alpha=0.0):
    if alpha ==0.0:
        lam = 1.0
    else: lam = np.random.beta(alpha, alpha)
    if indices is None: 
        indices = torch.randperm(inputs.size()[0]).to(inputs.device)
    images_b = inputs[indices]
    target_b = targets[indices]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    combined_images = inputs.detach().clone()
    combined_images[:, :, bbx1:bbx2, bby1:bby2] = images_b[:,
                                                            :, bbx1:bbx2, bby1:bby2]
    lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()
                                                    [-1] * inputs.size()[-2]))  # transformed ratio
    lam_a = torch.ones(inputs.shape[0], device=inputs.device)*lam_a
    lam_b = 1.-lam_a

    return combined_images, targets, target_b, lam_a, lam_b

def saliencymix(inputs, targets, indices=None, alpha=0.0):
    if indices is None: 
        indices = torch.randperm(inputs.size()[0]).to(inputs.device)
    if alpha ==0.0:
        lam = 1.0
    else: lam = np.random.beta(alpha, alpha)
    
    images_b = inputs[indices]
    target_b = targets[indices]

    bbx1, bby1, bbx2, bby2 = saliency_bbox(
        inputs[indices[0]], lam)
    combined_images = inputs.detach().clone()
    combined_images[:, :, bbx1:bbx2, bby1:bby2] = images_b[:,
                                                            :, bbx1:bbx2, bby1:bby2]

    lam_a = 1. - ((bbx2 - bbx1) * (bby2 - bby1) /
                    (inputs.size()[-1] * inputs.size()[-2]))
    lam_a = torch.ones(inputs.shape[0], device=inputs.device)*lam_a
    lam_b = 1.-lam_a

    return combined_images, targets, target_b, lam_a, lam_b

MIXUP_DICT={
    'mixup': input_mixup,
    'cutmix': cutmix,
    'saliencymix': saliencymix,
}
