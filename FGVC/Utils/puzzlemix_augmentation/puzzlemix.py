import os
import numpy as np
import torch
import torch.nn.functional as F
try:
    import gco
except:
    print("WARNING: you need to fix gco")
import time
from multiprocessing import Pool


def cost_matrix(width,device='cuda'):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)
    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i,m_j]= abs(i1-i2)**2 + abs(j1-j2)**2
    C = C/(width-1)**2
    C = torch.tensor(C).to(device)
    return C

def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]
    
    large_val = 1000 * block_num ** 2 
    
    if n_labels == 2:
        prior= eta * np.array([-np.log(alpha + 1e-8), -np.log(1 - alpha + 1e-8)]) / block_num ** 2
    elif n_labels == 3:
        prior= eta * np.array([-np.log(alpha**2 + 1e-8), -np.log(2 * alpha * (1-alpha) + 1e-8), -np.log((1 - alpha)**2 + 1e-8)]) / block_num ** 2
    elif n_labels == 4:
        prior= eta * np.array([-np.log(alpha**3 + 1e-8), -np.log(3 * alpha **2 * (1-alpha) + 1e-8), 
                             -np.log(3 * alpha * (1-alpha) **2 + 1e-8), -np.log((1 - alpha)**3 + 1e-8)]) / block_num ** 2
        
    unary_cost =  (large_val * np.stack([(1-lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(np.linspace(0,1, n_labels))], axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i-j)**2 / (n_labels-1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)

    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
    mask = labels.reshape(block_num, block_num)

    return mask

  
def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:,:,:-1,:] - input2[:,:,1:,:]
    pw_y = input1[:,:,:,:-1] - input2[:,:,:,1:]
    
    pw_x = pw_x[:,:,k-1::k,:]
    pw_y = pw_y[:,:,:,k-1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1,k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k,1))
    return pw_x, pw_y

def mixup_graph(input1,
                grad1,
                indices,
                block_num=2,
                alpha=0.5,
                beta=0.,
                gamma=0.,
                eta=0.2,
                neigh_size=2,
                n_labels=2,
                mean=None,
                std=None,
                transport=False,
                t_eps=10.0,
                t_size=-1,
                noise=None,
                adv_mask1=0,
                adv_mask2=0,
                cost_matrix_dict=None,
                device='cuda',
                get_only_mask=False,
                mp=None):
    '''Puzzle Mix'''
    input2 = input1[indices].clone()

    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    # if width == 64 or width ==32:
    neigh_size = min(neigh_size, block_size)
    if width ==224:# else: # 224
        neigh_size = min(neigh_size,2)
    t_size = min(t_size, block_size)

    # normalize
    beta = beta / block_num / 16

    # unary term
    if grad1.shape[-1] != block_size:
        grad1_pool = F.avg_pool2d(grad1, block_size)
    else:
        grad1_pool = grad1
    
    unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
    unary2_torch = unary1_torch[indices]

    # calculate pairwise terms
    if std is None and mean is None:
        input1_pool = F.avg_pool2d(input1, neigh_size)
    else:
        input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
    input2_pool = input1_pool[indices]

    pw_x = torch.zeros([batch_size, 2, 2, block_num - 1, block_num], device=device)
    pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num - 1], device=device)

    k = block_size // neigh_size
    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y

    # re-define unary and pairwise terms to draw graph
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    # solve graphcut
    if mp is None:
        mask = []
        for i in range(batch_size):
            mask.append(
                graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
    else:
        input_mp = []
        for i in range(batch_size):
            input_mp.append((unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
        mask = mp.starmap(graphcut_multi, input_mp)

    # optimal mask
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(1)

    # add adversarial noise
    if adv_mask1 == 1.:
        input1 = input1 * std + mean + noise
        input1 = torch.clamp(input1, 0, 1)
        input1 = (input1 - mean) / std

    if adv_mask2 == 1.:
        input2 = input2 * std + mean + noise[indices]
        input2 = torch.clamp(input2, 0, 1)
        input2 = (input2 - mean) / std

    # tranport
    if transport:
        if t_size == -1:
            t_block_num = block_num
            t_size = block_size
        elif t_size < block_size:
            # block_size % t_size should be 0
            t_block_num = width // t_size
            mask = F.interpolate(mask, size=t_block_num)
            grad1_pool = F.avg_pool2d(grad1, t_size)
            unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(
                batch_size, 1, 1)
            unary2_torch = unary1_torch[indices]
        else:
            t_block_num = block_num

        # input1
        plan1 = mask_transport(mask, unary1_torch,cost_matrix_dict, eps=t_eps)

        # input2
        plan2 = mask_transport(1 - mask, unary2_torch,cost_matrix_dict, eps=t_eps)

        if batch_size> 50:
            t_batch_size = 16
            try:
                for i in range((batch_size - 1) // t_batch_size + 1):
                    idx_from = i * t_batch_size
                    idx_to = min((i + 1) * t_batch_size, batch_size)
                    input1[idx_from:idx_to] = transport_image(input1[idx_from:idx_to],
                                                              plan1[idx_from:idx_to], block_num,
                                                              block_size)
                    input2[idx_from:idx_to] = transport_image(input2[idx_from:idx_to],
                                                              plan2[idx_from:idx_to], block_num,
                                                              block_size)
            except:
                raise AssertionError(
                    "** GPU memory is lacking while transporting. Reduce the t_batch_size value in this function (mixup.transprort) **"
                )
        else:
            input1 = transport_image(input1, plan1, t_block_num, t_size)
            input2 = transport_image(input2, plan2, t_block_num, t_size)

    # final mask and mixed ratio
    ratio = mask.reshape(batch_size, -1).mean(-1)
    mask = F.interpolate(mask, size=width).detach()
    if get_only_mask:
        return mask, ratio

    return mask * input1 + (1 - mask) * input2, ratio


def mask_transport(mask, grad_pool,cost_matrix_dict,eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]
    
    z = (mask>0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1-plan_win) * plan

        cost += plan_lose
    return plan_win
    

def transport_image(img, plan, block_num, block_size):
    '''apply transport plan to images'''
    batch_size = img.shape[0]
    C= img.shape[1]
    input_patch = img.reshape([batch_size, C, block_num, block_size, block_num * block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, C, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, C, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)

    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
    input_transport = input_transport.reshape([batch_size, C, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, C, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, C, block_num * block_size, block_num * block_size])
    
    return input_transport
