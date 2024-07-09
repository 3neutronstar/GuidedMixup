import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from Utils.comixup_augmentation.match import get_onehot_matrix, mix_input
import numpy as np
import os
from math import ceil


def mixup_process_worker(out: torch.Tensor,
                         target_reweighted: torch.Tensor,
                         hidden=0,
                         configs=None,
                         sc: torch.Tensor = None,
                         A_dist: torch.Tensor = None,
                         debug=False):
    """Perform Co-Mixup"""
    m_block_num = configs['m_block_num']
    n_input = out.shape[0]
    width = out.shape[-1]

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = width // m_block_num

    with torch.no_grad():
        if A_dist is None:
            A_dist = torch.eye(n_input, device=out.device)
        A_base = torch.eye(n_input, device=out.device)

        sc = F.avg_pool2d(sc, block_size)
        sc_norm = sc / sc.view(n_input, -1).sum(1).view(n_input, 1, 1)
        cost_matrix = -sc_norm

        A_dist = A_dist / torch.sum(A_dist) * n_input
        A = (1 - configs['m_omega']) * A_base + configs['m_omega'] * A_dist

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
                                        device=out.device)
        # Generate image and corrsponding soft target
        out, target_reweighted = mix_input(mask_onehot, out, target_reweighted)

    return out.contiguous(), target_reweighted


def mixup_process_worker_wrapper(q_input: mp.Queue, q_output: mp.Queue, device: int):
    """
    :param q_input:		input queue
    :param q_output:	output queue
    :param device:		running gpu device
    """
    #os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    print(f"Process generated with cuda:{device}")
    device = torch.device(f"cuda:{device}")
    while True:
        # Get and load on gpu
        out, target_reweighted, hidden, configs, sc, A_dist, debug = q_input.get()
        out = out.to(device)
        target_reweighted = target_reweighted.to(device)
        sc = sc.to(device)
        A_dist = A_dist.to(device)

        # Run
        out, target_reweighted = mixup_process_worker(out, target_reweighted, hidden, configs, sc,
                                                      A_dist, debug)
        # To cpu and return
        out = out.cpu()
        target_reweighted = target_reweighted.cpu()
        q_output.put([out, target_reweighted])


class MixupProcessWorker:
    def __init__(self, device: int):
        """
        :param device: gpu device id
        """
        self.q_input = mp.Queue()
        self.q_output = mp.Queue()
        self.worker = mp.Process(target=mixup_process_worker_wrapper,
                                 args=[self.q_input, self.q_output, device])
        self.worker.deamon = True
        self.worker.start()

    def start(self,
              out: torch.Tensor,
              target_reweighted: torch.Tensor,
              hidden=0,
              configs=None,
              sc: torch.Tensor = None,
              A_dist: torch.Tensor = None,
              debug=True):
        self.q_input.put([out, target_reweighted, hidden, configs, sc, A_dist, debug])

    def join(self):
        input, target = self.q_output.get()
        return input, target

    def close(self):
        self.worker.terminate()


class MixupProcessParallel:
    def __init__(self, part, batch_size, device_id=None):
        """
        :param part:
        :param batch_size:
        :param num_gpu:
        """
        self.part = part
        self.batch_size = batch_size
        self.n_workers = ceil(batch_size / part)
        if device_id:
            num_gpu=len(device_id)
            device_id=torch.arange(num_gpu)
        else:
            num_gpu=0
            device_id=[0]
        self.workers = [MixupProcessWorker(device=device_id[i % num_gpu]) for i in range(self.n_workers)]

    def __call__(self,
                 out: torch.Tensor,
                 target_reweighted: torch.Tensor,
                 hidden=0,
                 configs=None,
                 sc: torch.Tensor = None,
                 A_dist: torch.Tensor = None,
                 debug=False):
        '''
        :param out:					cpu tensor
        :param target_reweighted: 	cpu tensor
        :param hidden:
        :param configs:				cpu configs
        :param sc: 					cpu tensor
        :param A_dist: 				cpu tensor
        :param debug:
        :return:					out, target_reweighted (cpu tensor)
        '''
        batch_size=out.size(0)

        for idx in range(self.n_workers):
            if idx*self.part>=batch_size:
                break
            else:
                end_point=min((idx+1)*self.part, batch_size)
                self.workers[idx].start(
                    out[idx * self.part:end_point].contiguous(),
                    target_reweighted[idx * self.part:end_point].contiguous(), hidden, configs,
                    sc[idx * self.part:end_point].contiguous(),
                    A_dist[idx * self.part:end_point,
                        idx * self.part:end_point].contiguous(), debug)
        # join
        out_list = []
        target_list = []
        for idx in range(self.n_workers):
            if idx*self.part>=batch_size:
                break
            out, target = self.workers[idx].join()
            out_list.append(out)
            target_list.append(target)

        return torch.cat(out_list), torch.cat(target_list)

    def close(self):
        for w in self.workers:
            w.close()
