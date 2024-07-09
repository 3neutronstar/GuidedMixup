import numpy as np
import torch
import torch.nn.functional as F
import warnings
from Utils.comixup_augmentation.match import get_onehot_matrix, mix_input
from math import ceil

warnings.filterwarnings("ignore")


def mixup_process(out, target_reweighted, configs=None, sc=None, A_dist=None):
    m_block_num = configs['m_block_num']
    m_part = configs['m_part']
    batch_size = out.shape[0]
    width = out.shape[-1]

    if A_dist is None:
        A_dist = torch.eye(batch_size, device=out.device)

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)
    else:
        m_block_num = 2**int(m_block_num)

    block_size = width // m_block_num
    sc = F.avg_pool2d(sc, block_size)

    out_list = []
    target_list = []

    # Partition a batch
    for i in range(ceil(batch_size / m_part)):
        with torch.no_grad():
            sc_part = sc[i * m_part:(i + 1) * m_part]
            A_dist_part = A_dist[i * m_part:(i + 1) * m_part, i * m_part:(i + 1) * m_part]

            n_input = sc_part.shape[0]
            sc_norm = sc_part / sc_part.reshape(n_input, -1).sum(1).reshape(n_input, 1, 1)
            cost_matrix = -sc_norm

            A_base = torch.eye(n_input, device=out.device)
            A_dist_part = A_dist_part / torch.sum(A_dist_part) * n_input
            A = (1 - configs['m_omega']) * A_base + configs['m_omega'] * A_dist_part

            # Return a batch(partitioned) of mixup labeling
            mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                            A,
                                            n_output=n_input,
                                            beta=configs['m_beta'],
                                            gamma=configs['m_gamma'],
                                            eta=configs['m_eta'],
                                            mixup_alpha=configs['alpha'],
                                            thres=configs['m_thres'],
                                            thres_type=configs['m_thres_type'],
                                            set_resolve=configs['set_resolve'],
                                            niter=configs['m_niter'],
                                            device='cuda')

        # Generate image and corrsponding soft target
        output_part, target_part = mix_input(mask_onehot, out[i * m_part:(i + 1) * m_part],
                                             target_reweighted[i * m_part:(i + 1) * m_part])

        out_list.append(output_part)
        target_list.append(target_part)

    with torch.no_grad():
        out = torch.cat(out_list, dim=0)
        target_reweighted = torch.cat(target_list, dim=0)

    return out.contiguous(), target_reweighted