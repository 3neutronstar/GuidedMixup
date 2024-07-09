import sys
sys.path.append('.')
import torch
from Utils.correlation import cosine_similarity
import torch.nn as nn


def distance_function(a, b=None ,distance_metric='jsd'):
    ''' pytorch distance 
    input:
     - a: (batch_size1 N, n_features)
     - b: (batch_size2 M, n_features)
    output: NxM matrix'''
    if b is None:
        b=a
    if distance_metric=='cosine':
        distance = 1 - cosine_similarity(a.view(a.shape[0],-1), b.view(b.shape[0],-1))
    elif distance_metric=='cosine_abs':
        distance = 1 - cosine_similarity(a.view(a.shape[0],-1), b.view(b.shape[0],-1)).abs()
    elif distance_metric =='l1':
        ra = a.view(a.shape[0],-1).unsqueeze(1)
        rb = b.view(b.shape[0],-1).unsqueeze(0)
        distance = (ra-rb).abs().sum(dim=-1).view(a.shape[0],b.shape[0])
    elif distance_metric =='l2':
        ra = a.view(a.shape[0],-1).unsqueeze(1)
        rb = b.view(b.shape[0],-1).unsqueeze(0)
        distance = ((ra-rb).norm(dim=-1)).view(a.shape[0],b.shape[0])
    else:
        raise NotImplementedError
    return distance

    
if __name__ == '__main__':
    ps = torch.randn((4,224,224)).cpu()
    ps /= ps.sum(dim=[1, 2], keepdim=True)
    qs = ps.detach().clone()
    import time
    print(distance_function(ps,qs,'l2'))
    # qs = torch.randn((40,224,224)).cuda()
    # jsdiv = JSDiv()
    # y = jsdiv(ps, qs)
    # print(y)
    # print(x.size())
    # print(x)
    # x = tv_dist(ps, qs)
    # print(x.size())
    # print(x)


    jsdiv_tik=time.time()
    distance_function(ps,qs,'jsd').cpu()
    jsdiv_tok=time.time()

    l1_tik=time.time()
    dis=distance_function(ps,qs,'l1').cpu()
    print(dis)
    l1_tok=time.time()

    l2_tik=time.time()
    dis=distance_function(ps,qs,'l2').cpu()
    l2_tok=time.time()

    cosine_tik=time.time()
    distance_function(ps,qs,'cosine').cpu()
    cosine_tok=time.time()

    print('jsdiv_dist:{:.5f}, l1: {:.5f}, l2: {:.5f}, cosine: {:.5f}'.format(jsdiv_tok-jsdiv_tik,l1_tok-l1_tik,l2_tok-l2_tik,cosine_tok-cosine_tik))
