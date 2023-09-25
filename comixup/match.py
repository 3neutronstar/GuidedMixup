import torch
import torch.nn.functional as F
import gco
import numpy as np
import itertools
import time


def to_onehot(idx, n_input, device='cuda'):
    '''Return one-hot vector'''
    idx_onehot = torch.zeros((idx.shape[0], n_input), dtype=torch.float32, device=device)
    idx_onehot.scatter_(1, idx.unsqueeze(1), 1)
    return idx_onehot


def random_initialize(n_input, n_output, height, width):
    '''Initialization of labeling for Co-Mixup'''
    return np.random.randint(0, n_input, (n_output, width, height))


def obj_fn(cost_matrix, mask_onehot, beta, gamma):
    '''Calculate objective without thresholding'''
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)
    loss += beta / 2 * (((mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :])**2).sum() +
                        ((mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :])**2).sum())
    loss += gamma * (torch.sum(mask_idx_sum.sum(0)**2) - torch.sum(mask_idx_sum**2))

    return loss


def obj_fn_thres(cost_matrix, mask_onehot, beta, gamma, thres):
    '''Calculate objective with thresholding'''
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)
    loss += beta / 2 * (((mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :])**2).sum() +
                        ((mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :])**2).sum())

    penalty = mask_idx_sum.sum(0, keepdim=True) - mask_idx_sum
    modular_penalty = (penalty > thres).float() * penalty
    loss += gamma * torch.sum(modular_penalty * mask_idx_sum)

    return loss


def mix_input(mask_onehot, input_sp, target_reweighted, sc=None):
    ''' Mix inputs and one-hot labels based on labeling (mask_onehot)'''
    n_output, height, width, n_input = mask_onehot.shape
    _, n_class = target_reweighted.shape

    mask_onehot_im = F.interpolate(mask_onehot.permute(0, 3, 1, 2),
                                   size=input_sp.shape[-1],
                                   mode='nearest')
    output = torch.sum(mask_onehot_im.unsqueeze(2) * input_sp.unsqueeze(0), dim=1)

    if sc is None:
        mask_target = torch.matmul(mask_onehot, target_reweighted)
    else:
        weighted_mask = mask_onehot * sc.permute(1, 2, 0).unsqueeze(0)
        mask_target = torch.matmul(weighted_mask, target_reweighted)

    target = mask_target.reshape(n_output, height * width, n_class).sum(-2)
    target /= target.sum(-1, keepdim=True)

    return output, target


def resolve_label(assigned_label_total, device='cuda'):
    '''A post-processing for resolving identical outputs'''
    n_output, n_input = assigned_label_total.shape
    add_cost = torch.zeros_like(assigned_label_total)

    dist = torch.min(
        (assigned_label_total.unsqueeze(1) - assigned_label_total.unsqueeze(0)).abs().sum(-1),
        torch.tensor(1.0, device=device))
    coincide = torch.triu(1. - dist, diagonal=1)

    for i1, i2 in coincide.nonzero():
        nonzeros = assigned_label_total[i1].nonzero()
        if len(nonzeros) == 1:
            continue
        else:
            add_cost[i1][nonzeros[0]] = 1.
            add_cost[i2][nonzeros[1]] = 1.

    return add_cost


def graphcut_multi(cost, beta=1, algorithm='swap', n_label=0, add_idx=None):
    '''find optimal labeling using Graph-Cut algorithm'''
    height, width, n_input = cost.shape

    unary = np.ascontiguousarray(cost)
    pairwise = (np.ones(shape=(n_input, n_input), dtype=np.float32) -
                np.eye(n_input, dtype=np.float32))
    if n_label == 2:
        pairwise[-1, :-1][add_idx] = 0.25
        pairwise[:-1, -1][add_idx] = 0.25
    elif n_label == 3:
        pairwise[-3:, :-3][:, add_idx] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])
        pairwise[:-3, -3:][add_idx, :] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])

    cost_v = beta * np.ones(shape=[height - 1, width], dtype=np.float32)
    cost_h = beta * np.ones(shape=[height, width - 1], dtype=np.float32)

    mask_idx = gco.cut_grid_graph(unary, pairwise, cost_v, cost_h, algorithm=algorithm)
    return mask_idx


def graphcut_wrapper(cost_penalty, label_count, n_input, height, width, beta, device, iter_idx=0):
    '''Wrapper of graphcut_multi performing efficient extension to multi-label'''
    assigned_label = (label_count > 0)
    if iter_idx > 0:
        n_label = int(assigned_label.float().sum())
    else:
        n_label = 0

    if n_label == 2:
        cost_add = cost_penalty[:, :, assigned_label].mean(-1, keepdim=True) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=2,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 1,
                                    device=device).reshape(height, width, n_input + 1)

        idx_matrix = torch.zeros([1, 1, n_input], device=device)
        idx_matrix[:, :, assigned_label] = 0.5
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + mask_idx_onehot[:, :,
                                                                          n_input:] * idx_matrix
    elif n_label >= 3:
        soft_label = torch.tensor([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], device=device)

        _, indices = torch.topk(label_count, k=3)
        assigned_label = torch.zeros_like(assigned_label)
        assigned_label[indices] = True

        cost_add = torch.matmul(cost_penalty[:, :, assigned_label], soft_label) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=3,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 3,
                                    device=device).reshape(height, width, n_input + 3)

        idx_matrix = torch.zeros([3, n_input], device=device)
        idx_matrix[:, assigned_label] = soft_label
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + torch.matmul(
            mask_idx_onehot[:, :, n_input:], idx_matrix)
    else:
        unary = cost_penalty.cpu().numpy()
        mask_idx_np = graphcut_multi(unary, beta=beta, algorithm='swap')
        mask_onehot_i = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                  n_input,
                                  device=device).reshape(height, width, n_input)

    return mask_onehot_i


def get_onehot_matrix(cost_matrix,
                      A,
                      n_output,
                      idx=None,
                      beta=0.32,
                      gamma=1.,
                      eta=0.05,
                      mixup_alpha=2.0,
                      thres=0.84,
                      thres_type='hard',
                      set_resolve=True,
                      niter=3,
                      device='cuda'):
    '''Iterative submodular minimization algorithm with the modularization of supermodular term'''
    n_input, height, width = cost_matrix.shape
    thres = thres * height * width
    beta = beta / height / width
    gamma = gamma / height / width
    eta = eta / height / width

    add_cost = None

    # Add prior term
    lam = mixup_alpha * torch.ones(n_input, device=device)
    alpha = torch.distributions.dirichlet.Dirichlet(lam).sample().reshape(n_input, 1, 1)
    cost_matrix -= eta * torch.log(alpha + 1e-8)

    with torch.no_grad():
        # Init
        if idx is None:
            mask_idx = torch.tensor(random_initialize(n_input, n_output, height, width),
                                    device=device)
        else:
            mask_idx = idx

        mask_onehot = to_onehot(mask_idx.reshape(-1), n_input,
                                device=device).reshape([n_output, height, width, n_input])

        loss_prev = obj_fn(cost_matrix, mask_onehot, beta, gamma)
        penalty = to_onehot(mask_idx.reshape(-1), n_input, device=device).sum(0).reshape(-1, 1, 1)

        # Main loop
        for iter_idx in range(niter):
            for i in range(n_output):
                label_count = mask_onehot[i].reshape([height * width, n_input]).sum(0)
                penalty -= label_count.reshape(-1, 1, 1)
                if thres_type == 'hard':
                    modular_penalty = (2 * gamma * (
                        (A @ penalty.squeeze() > thres).float() * A @ penalty.squeeze())).reshape(
                            -1, 1, 1)
                elif thres_type == 'soft':
                    modular_penalty = (2 * gamma * ((A @ penalty.squeeze() > thres).float() *
                                                    (A @ penalty.squeeze() - thres))).reshape(
                                                        -1, 1, 1)
                else:
                    raise AssertionError("wrong threshold type!")

                if add_cost is not None:
                    cost_penalty = (cost_matrix + modular_penalty +
                                    gamma * add_cost[i].reshape(-1, 1, 1)).permute(1, 2, 0)
                else:
                    cost_penalty = (cost_matrix + modular_penalty).permute(1, 2, 0)

                mask_onehot[i] = graphcut_wrapper(cost_penalty, label_count, n_input, height, width,
                                                  beta, device, iter_idx)
                penalty += mask_onehot[i].reshape([height * width,
                                                   n_input]).sum(0).reshape(-1, 1, 1)

            if iter_idx == niter - 2 and set_resolve:
                assigned_label_total = (mask_onehot.reshape(n_output, -1, n_input).sum(1) >
                                        0).float()
                add_cost = resolve_label(assigned_label_total, device=device)

            loss = obj_fn(cost_matrix, mask_onehot, beta, gamma)
            if (loss_prev - loss).abs() / loss.abs() < 1e-6:
                break
            loss_prev = loss

    return mask_onehot


if __name__ == '__main__':
    # Some test for debugging and computation time
    import time
    import argparse

    def brute_force(cost_matrix, n_output, beta, gamma, idx_list=None):
        n_input, n_block, _ = cost_matrix.shape

        # Brute force
        loss_min_bf = 1000000
        for k, idx in enumerate(idx_list):
            z = np.zeros((n_output * n_block**2, n_input))  # one-hot representation of output
            for i in range(n_output * n_block**2):
                z[i, idx[i]] = 1

            mask_onehot = torch.tensor(z.reshape(n_output, n_block, n_block, n_input),
                                       device=cost_matrix.device)
            loss = obj_fn(cost_matrix, mask_onehot, beta / n_block**2, gamma / n_block**2)

            if loss < loss_min_bf:
                loss_min_bf = loss

            if k % 1000 == 0:
                print(k, end='\r')

        return loss_min_bf

    parser = argparse.ArgumentParser(description='Algorithm Test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_output', type=int, default=100)
    parser.add_argument('--n_input', type=int, default=100)
    parser.add_argument('--n_block', type=int, default=8)
    parser.add_argument('--n_part', type=int, default=20)
    parser.add_argument('--n_iter', type=int, default=4)
    parser.add_argument('--solver', type=str, default='gc', choices=['gc'])
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    beta = 0.8
    gamma = 1.0
    eta = 0.0
    device = 'cuda'

    n_input = args.n_input
    n_output = args.n_output
    n_block = args.n_block

    obj_list = []
    obj_rand_list = []
    obj_exact_list = []

    a = list(range(n_input))
    b = [a for _ in range(n_output * n_block**2)]
    idx_list = list(itertools.product(*b))  # all possible combination of outputs
    print("start brute search: ", len(idx_list))

    for _ in range(100):
        input_sp = torch.from_numpy(np.random.normal(0, 1,
                                                     size=(n_input, 3, 64, 64))).cuda().float()
        target_reweighted = torch.eye(n_input).cuda().float()

        cost_matrix = torch.from_numpy(np.random.normal(0, 1,
                                                        size=(n_input, n_block,
                                                              n_block))).to(device).abs().float()
        cost_matrix = cost_matrix  #/ cost_matrix.view(n_input, -1).sum(1).view(n_input, 1, 1)
        A = torch.eye(n_input, device=device)

        output_list = []
        target_list = []
        n_part = min(args.n_part, n_output)
        s_init = time.time()

        unary_first = []
        unary_last = []
        loss_first = []
        loss_last = []

        for i in range(n_output // n_part):
            mask_onehot = get_onehot_matrix(cost_matrix[i * n_part:(i + 1) * n_part].detach(),
                                            A[i * n_part:(i + 1) * n_part,
                                              i * n_part:(i + 1) * n_part],
                                            n_part,
                                            beta=beta,
                                            gamma=gamma,
                                            eta=eta,
                                            thres=0.82,
                                            device=device,
                                            niter=args.n_iter)
            # print("gc time: {}".format(time.time()-s))

            # s = time.time()
            output, target = mix_input(mask_onehot,
                                       input_sp[i * n_part:(i + 1) * n_part],
                                       target_reweighted[i * n_part:(i + 1) * n_part],
                                       sc=-cost_matrix[i * n_part:(i + 1) * n_part])
            # print((target>0).float().sum(-1).mean(0))
            # print("mixup time: {}".format(time.time()-s))
            obj = obj_fn(cost_matrix, mask_onehot, beta / n_block**2, gamma / n_block**2)
            obj_list.append(obj)
            print("obj (our): {:.3f}".format(obj))

            mask_onehot_rand = torch.bernoulli(torch.zeros_like(mask_onehot) + 0.5)
            obj_rand = obj_fn(cost_matrix, mask_onehot_rand, beta / n_block**2, gamma / n_block**2)
            obj_rand_list.append(obj_rand)
            print("obj (rand): {:.3f}".format(obj_rand))

            obj_exact = cost_matrix.sum()
            # obj_exact = brute_force(cost_matrix, n_output, beta, gamma, idx_list)
            obj_exact_list.append(obj_exact)
            print("obj (exact): {:.3f}\n".format(obj_exact))

            output_list.append(output)
            target_list.append(target)

        output = torch.cat(output_list, dim=0)
        target = torch.cat(target_list, dim=0)

    obj_mean = torch.mean(torch.tensor(obj_list))
    obj_rand_mean = torch.mean(torch.tensor(obj_rand_list))
    obj_exact_mean = torch.mean(torch.tensor(obj_exact_list))

    print("obj (our): {:.3f}".format(obj_mean))
    print("obj (rand): {:.3f}".format(obj_rand_mean))
    print("obj (exact): {:.3f}".format(obj_exact_mean))
    print("rel ratio: {:.3f}".format(
        (obj_mean - obj_exact_mean) / (obj_rand_mean - obj_exact_mean)))
    print("\ntotal time: {}".format(time.time() - s_init))