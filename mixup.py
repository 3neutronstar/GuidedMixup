import numpy as np
import torch
import torch.nn.functional as F
import gco
import math
import torchvision.transforms.functional as TF
from comixup.match import mix_input, get_onehot_matrix
import torch.nn as nn
try:
    import sys
    from pairing import onecycle_cover
except: pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cosine_similarity(a, b):
    dot = a.matmul(b.t())
    norm =a.norm(dim=1,keepdim=True).matmul(b.norm(dim=1,keepdim=True).t())
    return dot / norm

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

def pairing(sc_a, sc_b=None,condition='random',distance_metric='l2'):
        '''
        - skipping - 
        if idx > images.size(0)*0.8:
            sorted_indices[row]=row                                    
            # non_skip_idx.append(row) # if you don't select, then delete it
        else: 
            sorted_indices[row]=col
            # non_skip_idx.append(row)
        '''
        if sc_b is None:
            sc_b=sc_a

        if condition == 'greedy':
                X = distance_function(
                    sc_a, sc_b, distance_metric).cpu().numpy()
                sorted_indices=onecycle_cover(X)
        else:  # random
            sorted_indices = np.random.permutation(sc_a.size(0))
        return sorted_indices

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


def cost_matrix(width, device='cuda'):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)

    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i, m_j] = abs(i1 - i2)**2 + abs(j1 - j2)**2

    C = C / (width - 1)**2
    C = torch.tensor(C)
    if device == 'cuda':
        C = C.cuda()

    return C


cost_matrix_dict = {
    '2': cost_matrix(2, device).unsqueeze(0),
    '4': cost_matrix(4, device).unsqueeze(0),
    '8': cost_matrix(8, device).unsqueeze(0),
    '16': cost_matrix(16, device).unsqueeze(0),
    '32': cost_matrix(32, device).unsqueeze(0),
}


def mixup_process(out,
                  target_reweighted,
                  hidden=0,
                  args=None,
                  grad=None,
                  noise=None,
                  adv_mask1=0,
                  adv_mask2=0,
                  mp=None,
                  A_dist=None,
                  align=False):
    '''various mixup process'''
    if args is not None:
        mixup_alpha = args.mixup_alpha
        in_batch = args.in_batch
        mean = args.mean
        std = args.std
        box = args.box
        graph = args.graph
        beta = args.beta
        gamma = args.gamma
        eta = args.eta
        neigh_size = args.neigh_size
        n_labels = args.n_labels
        transport = args.transport
        t_eps = args.t_eps
        t_size = args.t_size
        guided = args.guided
        condition=args.condition
        comixup=args.comixup
        block_num=args.block_num
        dsr=1
    ratio_b=None
    if block_num is None:
        block_num = 2**np.random.randint(1, 5)
    indices = np.random.permutation(out.size(0))
    if mixup_alpha is None:
        lam=0
    else:
        lam = get_lambda(mixup_alpha)

    if hidden:
        # Manifold Mixup
        out = out * lam + out[indices] * (1 - lam)
        ratio = torch.ones(out.shape[0], device='cuda') * lam
    else:
        if comixup:
            sc=grad
            m_block_num = args.m_block_num
            m_part = args.m_part
            batch_size = out.shape[0]
            width = out.shape[-1]

            if A_dist is None:
                A_dist = torch.eye(batch_size, device=out.device)

            if m_block_num == -1:
                m_block_num = 2**np.random.randint(1, 5)

            block_size = width // m_block_num
            sc = F.avg_pool2d(sc, block_size)

            out_list = []
            target_list = []

            # Partition a batch
            for i in range(math.ceil(batch_size / m_part)):
                with torch.no_grad():
                    sc_part = sc[i * m_part:(i + 1) * m_part]
                    A_dist_part = A_dist[i * m_part:(i + 1) * m_part, i * m_part:(i + 1) * m_part]

                    n_input = sc_part.shape[0]
                    sc_norm = sc_part / sc_part.reshape(n_input, -1).sum(1).reshape(n_input, 1, 1)
                    cost_matrix = -sc_norm

                    A_base = torch.eye(n_input, device=out.device)
                    A_dist_part = A_dist_part / torch.sum(A_dist_part) * n_input
                    A = (1 - args.m_omega) * A_base + args.m_omega * A_dist_part

                    # Return a batch(partitioned) of mixup labeling
                    mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                                    A,
                                                    n_output=n_input,
                                                    beta=args.m_beta,
                                                    gamma=args.m_gamma,
                                                    eta=args.m_eta,
                                                    mixup_alpha=args.mixup_alpha,
                                                    thres=args.m_thres,
                                                    thres_type=args.m_thres_type,
                                                    set_resolve=args.set_resolve,
                                                    niter=args.m_niter,
                                                    device='cuda')

                # Generate image and corrsponding soft target
                output_part, target_part = mix_input(mask_onehot, out[i * m_part:(i + 1) * m_part],
                                                    target_reweighted[i * m_part:(i + 1) * m_part])

                out_list.append(output_part)
                target_list.append(target_part)

            with torch.no_grad():
                out = torch.cat(out_list, dim=0).contiguous()
                target_reweighted = torch.cat(target_list, dim=0)
            return out, target_reweighted
        elif box:
            # CutMix
            out, ratio = mixup_box(out, out[indices], alpha=lam, device='cuda')
        elif graph:
            # PuzzleMix
            if block_num >= 1:
                out, ratio = mixup_graph(out,
                                         grad,
                                         indices,
                                         block_num=block_num,
                                         alpha=lam,
                                         beta=beta,
                                         gamma=gamma,
                                         eta=eta,
                                         neigh_size=neigh_size,
                                         n_labels=n_labels,
                                         mean=mean,
                                         std=std,
                                         transport=transport,
                                         t_eps=t_eps,
                                         t_size=t_size,
                                         noise=noise,
                                         adv_mask1=adv_mask1,
                                         adv_mask2=adv_mask2,
                                         mp=mp,
                                         device='cuda')
            else:
                ratio = torch.ones(out.shape[0], device='cuda')
        elif guided:
            sc_a=grad
            # guidedmixup
            sc_a = TF.gaussian_blur(sc_a,(7,7),(3,3))
            sc_a /= (sc_a).sum(dim=[-1, -2], keepdim=True)
            if condition=='greedy':
                try:
                    indices = pairing(sc_a.detach(),condition='greedy_c',distance_metric='l2')
                except:
                    indices = pairing(sc_a.detach(),condition=condition,distance_metric='l2')
            else:
                indices = pairing(sc_a.detach(),condition=condition,distance_metric='l2')

            out_b = out[indices]
            sc_b = sc_a[indices]
            norm_sc_a = torch.div(sc_a, (sc_a+sc_b).detach())
            ratio=norm_sc_a.mean(dim=[-1, -2]).unsqueeze(-1)
            mask_a = torch.stack([norm_sc_a]*3, dim=1)
            out = mask_a*out + (1-mask_a)*out_b
            ratio_b=1.-ratio
        else:
            # Input Mixup
            out = out * lam + out[indices] * (1 - lam)
            ratio = torch.ones(out.shape[0], device='cuda') * lam

    target_shuffled_onehot = target_reweighted[indices]
    if ratio.dim() == 1:
        ratio = ratio.unsqueeze(-1)
    if ratio_b is None:
        target_reweighted = target_reweighted * ratio + target_shuffled_onehot * (
            1 - ratio)
    else:
        target_reweighted = target_reweighted * ratio + target_shuffled_onehot * (
            ratio_b)
    # save "out" in image format
    # torchvision.utils.save_image(out, 'out.png',normalize=True)
    # print("SAVE")
    # print(time.time()-tik)
    return out, target_reweighted


def get_lambda(alpha=1.0, alpha2=None):
    '''Return lambda'''
    if alpha > 0.:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
    else:
        lam = 1.
    return lam


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, eps=1e-8):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]

    large_val = 1000 * block_num**2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array([
            -np.log(alpha**2 + eps), -np.log(2 * alpha * (1 - alpha) + eps),
            -np.log((1 - alpha)**2 + eps)
        ])
    elif n_labels == 4:
        prior = np.array([
            -np.log(alpha**3 + eps), -np.log(3 * alpha**2 * (1 - alpha) + eps),
            -np.log(3 * alpha * (1 - alpha)**2 + eps), -np.log((1 - alpha)**3 + eps)
        ])

    prior = eta * prior / block_num**2
    unary_cost = (large_val * np.stack([(1 - lam) * unary1 + lam * unary2 + prior[i]
                                        for i, lam in enumerate(np.linspace(0, 1, n_labels))],
                                       axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j)**2 / (n_labels - 1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y,
                                      algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask


def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

    return pw_x, pw_y


def mixup_box(input1, input2, alpha=0.5, device='cuda'):
    '''CutMix'''
    batch_size, _, height, width = input1.shape
    ratio = np.zeros([batch_size])

    rx = np.random.uniform(0, height)
    ry = np.random.uniform(0, width)
    rh = np.sqrt(1 - alpha) * height
    rw = np.sqrt(1 - alpha) * width
    x1 = int(np.clip(rx - rh / 2, a_min=0., a_max=height))
    x2 = int(np.clip(rx + rh / 2, a_min=0., a_max=height))
    y1 = int(np.clip(ry - rw / 2, a_min=0., a_max=width))
    y2 = int(np.clip(ry + rw / 2, a_min=0., a_max=width))
    input1[:, :, x1:x2, y1:y2] = input2[:, :, x1:x2, y1:y2]
    ratio += 1 - (x2 - x1) * (y2 - y1) / (width * height)

    ratio = torch.tensor(ratio, dtype=torch.float32)
    if device == 'cuda':
        ratio = ratio.cuda()

    return input1, ratio


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
                t_size=16,
                noise=None,
                adv_mask1=0,
                adv_mask2=0,
                device='cuda',
                mp=None):
    '''Puzzle Mix'''
    input2 = input1[indices].clone()

    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
    t_size = min(t_size, block_size)

    # normalize
    beta = beta / block_num / 16

    # unary term
    grad1_pool = F.avg_pool2d(grad1, block_size)
    unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
    unary2_torch = unary1_torch[indices]

    # calculate pairwise terms
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
    mask = torch.tensor(np.array(mask), dtype=torch.float32, device=device)
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
        plan = mask_transport(mask, unary1_torch, eps=t_eps)
        input1 = transport_image(input1, plan, batch_size, t_block_num, t_size)

        # input2
        plan = mask_transport(1 - mask, unary2_torch, eps=t_eps)
        input2 = transport_image(input2, plan, batch_size, t_block_num, t_size)

    # final mask and mixed ratio
    mask = F.interpolate(mask, size=width)
    ratio = mask.reshape(batch_size, -1).mean(-1)

    return mask * input1 + (1 - mask) * input2, ratio


def mask_transport(mask, grad_pool, eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]

    z = (mask > 0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)

    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1 - plan_win) * plan

        cost += plan_lose

    return plan_win


def transport_image(img, plan, batch_size, block_num, block_size):
    '''apply transport plan to images'''
    input_patch = img.reshape([batch_size, 3, block_num, block_size,
                               block_num * block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size,
                                       block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size,
                                       block_size]).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    input_transport = plan.transpose(
        -2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(
            0, 1, 4, 2, 3)
    input_transport = input_transport.reshape(
        [batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num * block_size, block_num * block_size])

    return input_transport
