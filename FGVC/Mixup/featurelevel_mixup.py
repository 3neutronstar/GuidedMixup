from Mixup.mixup import cutmix, input_mixup
from Utils.sinkhorn_distance import SinkhornDistance
import numpy as np
import random
import torch
import torch.nn.functional as F


############################################
############ manifold mixup ################
############################################


def manifold_mixup(feature, targets, indices=None,alpha=0.0):
    if alpha < 0:
        mixup_lam = 0.5
    else:
        mixup_lam = np.random.beta(alpha, alpha)
    if indices is None:
        indices = np.random.permutation(feature.shape[0])

    targets_b = targets[indices]
    feature_b = feature[indices]
    mixup_feature = mixup_lam*feature + (1-mixup_lam)*feature_b

    return mixup_feature, targets, targets_b, mixup_lam

############################################
############## align mixup #################
############################################


def alignmixup(feature, targets, indices=None,alpha=0.0):
    mixup_lam = np.random.beta(alpha, alpha)
    # out shape = batch_size x 512 x 4 x 4 (cifar10/100)
    if indices is None:
        indices = np.random.permutation(feature.size(0))
    # batch_size x 512 x 16
    feat1 = feature.view(feature.shape[0], feature.shape[1], -1)
    feat2 = feature[indices].view(
        feature.shape[0], feature.shape[1], -1)  # batch_size x 512 x 16

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    P = sinkhorn(feat1.permute(0, 2, 1), feat2.permute(
        0, 2, 1)).detach()  # optimal plan batch x 16 x 16

    P = P*(feature.size(2)*feature.size(3))  # assignment matrix

    # uniformly choose at random, which alignmix to perform
    align_mix = random.randint(0, 1)

    if (align_mix == 0):
        # \tilde{A} = A'R^{T}
        f1 = torch.matmul(feat2, P.permute(0, 2, 1).cuda()).view(feature.shape)
        final = feat1.view(feature.shape)*mixup_lam + f1*(1-mixup_lam)

    elif (align_mix == 1):
        # \tilde{A}' = AR
        f2 = torch.matmul(feat1, P.cuda()).view(feature.shape).cuda()
        final = f2*mixup_lam + feat2.view(feature.shape)*(1-mixup_lam)

    targets, targets_b = targets, targets[indices]

    return final, targets, targets_b, mixup_lam


def mixup_process(x, feature, targets, layer_mix, sc=None, mixup_indices=None, configs=None, verboses=None):
    '''
        input
        x: input image
        feature: feature map
        targets: label
        layer_mix: layer index
        sc: saliency score from the model
        configs: configs
    '''

    verboses.update({'mask_track': None, 'norm': {},'mixup_loc':layer_mix})

    mixup_lam_b = None
    if configs['train_mode']=='manifoldmixup':
        mixup_feature, targets, targets_b, mixup_lam = manifold_mixup(feature, targets, mixup_indices, configs['alpha'])
    elif configs['train_mode']=='alignmixup':
        if layer_mix==0: # input level
            mixup_feature,targets,targets_b,mixup_lam,mixup_lam_b= input_mixup(feature, targets, mixup_indices, configs['alpha'])
        else: # embedding level
            mixup_feature, targets, targets_b, mixup_lam = alignmixup(feature, targets, mixup_indices, configs['alpha'])
    else:
        raise ValueError('Unknown train_mode {}'.format(configs['train_mode']))

    if mixup_lam_b is None:
        mixup_lam_b = 1. - mixup_lam
    return mixup_feature, targets, targets_b, mixup_lam, mixup_lam_b, verboses
