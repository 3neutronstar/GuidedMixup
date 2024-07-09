import numpy as np
import torch
import torch.nn.functional as F
import cv2

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise NotImplementedError
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

@torch.no_grad()
def get_spm(input, target, imgsize, model,model_name=None):

    bs = input.size(0)
    with torch.no_grad():
        _, fms = model(input,extract_feature=True)
        if 'preact_resnet' in model_name:
            fms = fms[-1]
            clsw = model.module.linear
        elif 'resnet' in model_name:
            fms = fms[-1]
            clsw = model.module.fc  # need to check name
        elif 'densenet' in model_name:
            fms = fms[-1]
            clsw = model.module.linear
        elif 'mobilenetv2' in model_name:
            fms = fms[-1]
            clsw = model.module.linear
        else:
            raise NotImplementedError('not support this backbone! Please designate classifier name to this function')
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea), dim=1)
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(
                0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(
                outmaps, imgsize, mode='bilinear', align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps, clslogit

def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(
        np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
