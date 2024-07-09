import copy
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from Utils.distance import distance_function
from Utils.normalize_denormalize import minmax_norm
from Utils.spectral_residual import SpectralResidual
import cv2
import torch.nn.functional as F
from multiprocessing import Pool
try:
    from pairing import onecycle_cover
except: pass


def multi_greedy(Xs, start,skip_prob=0.0):  # multiprocessing version of final guidedmixup
    # x = Xs[start:end, start:end]
    sorted_indices = -1*np.ones(Xs.shape[0], dtype=int)
    max_idx = np.argmax(Xs)
    row, col = max_idx//Xs.shape[0], max_idx % Xs.shape[0]
    first_row = copy.deepcopy(row)
    idx = 0
    sorted_indices[row] = col
    # print(row, col)
    Xs[:,row]=0
    while idx < Xs.shape[0] - 2:
        idx += 1
        row = col
        col = np.argmax(Xs[row])
        sorted_indices[row] = col
        Xs[:,row] = 0
    sorted_indices[col] = first_row
    # print(sorted_indices,'a')
    return sorted_indices+start


class GuidedMixupLearner(Baseline):
    condition_func = {
        'greedy': multi_greedy,
    }

    def __init__(self, model, file_name, save_path, device, configs):
        super(GuidedMixupLearner, self).__init__(
            model, file_name, save_path, device, configs)
        if self.configs['blur']>0:
            if 'blur_sigma_random' in self.configs.keys():
                self.blur = transforms.GaussianBlur((self.configs['blur'],self.configs['blur']), (0.5, 3))
            else:
                self.blur = transforms.GaussianBlur((self.configs['blur'], self.configs['blur']), sigma=(self.configs['blur_sigma'], self.configs['blur_sigma']))  # change this
        else:
            self.blur = None

        if 'sr' in self.configs['train_mode']:
            self.saliency = SpectralResidual(
                self.configs['saliency_blur'], self.configs['saliency_blur_sigma'], device=self.device)
            # self.configs['blur'], self.configs['blur_sigma'], device=self.device)

        if self.configs['condition'] not in ['greedy_c','random'] and self.configs['batch_size'] > self.configs['m_part']:
            if self.configs['m_part'] < self.configs['batch_size']:
                self.n_proc = self.configs['batch_size']//self.configs['m_part']
                self.start_idx = [i*self.configs['m_part']
                                  for i in range(self.n_proc)]
                self.end_idx = []
                error_count = 0
                for i in range(self.n_proc):
                    if (i+1)*self.configs['m_part'] <= self.configs['batch_size']:
                        self.end_idx.append((i+1)*self.configs['m_part'])
                    else:
                        self.end_idx.append(self.configs['batch_size'])
                        if error_count == 1:
                            raise NotImplementedError
                        error_count += 1
                if self.configs['condition']=='bruteforce_pair':
                    self.start_idx.append(self.configs['m_part']*self.n_proc)
                else:
                    self.pool = Pool(self.n_proc)

    def _before_train(self, train_loader, valid_loader):
        super()._before_train(train_loader, valid_loader)


    def _saliency_normalization(self, sc, normalization_type='sumto1'):
        # sc: saliency map
        # normalization_type: sumto1, tanh, softmax, mean_tanh, minmax, none
        # output -> normalized saliency: sc
        if normalization_type == 'sumto1':
            sc /= (sc).sum(dim=[-1, -2], keepdim=True) 
        else:
            raise NotImplementedError('saliency_normalization \"{}\" is not implemented'.format(normalization_type))

        return sc

    def _train(self, loader, epoch,finetune=False):
        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        if self.configs['training_verbose']:
            progress = ProgressMeter(
                len(loader), [batch_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))
        else:
            progress = ProgressMeter(
                len(loader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))
        self.model.train()
        end = time.time()
        tik = time.time()
        i = 0
        once = True

        for minibatch in loader:
            if len(minibatch)==2:
                images, targets = minibatch
            else:
                images, targets,_=minibatch
            images, targets = images.to(self.device), targets.to(self.device)
            B, C, H, W = images.size()

            r = np.random.rand(1)
            # if self.configs['train_mode']=='guided-ap':
            #     self.optimizer.zero_grad()
            if r < self.configs['mix_prob'] and self.configs['mixup_epochs']< epoch and not finetune:
                target_a = targets
                images_a = images
                sc_a = []
                # extract saliency #
                if self.configs['train_mode']=='guided-sr':
                    sc_a = self.saliency.transform_spectral_residual(images_a) # +1e-12
                elif self.configs['train_mode'] == 'guided-ap':
                    self.optimizer.zero_grad()
                    images_a = images.clone()
                    images_a.requires_grad = True
                    if self.configs['clean_lam'] == 0:
                        loss, outputs = self._forward_all(images_a, targets)
                        self._just_backward(loss,retain_graph=False)
                    else:
                        loss, outputs = self._forward_all(images_a, targets)
                        loss*=self.configs['clean_lam']
                        self._just_backward(loss,retain_graph=True)
                    # sc_a = (images_a.grad+1e-12).norm(dim=1).detach().clone()
                    sc_a = (((images_a.grad)**2).mean(dim=1)).sqrt().detach() # +1e-12

                    if self.configs['clean_lam'] == 0:
                        self.model.train()
                        self.optimizer.zero_grad()
                else:
                    raise NotImplementedError

                if self.blur:
                    sc_a = self.blur(sc_a)
                else:
                    sc_a = sc_a
                
                sc_a=self._saliency_normalization(sc_a, self.configs['saliency_normalization'])
                # pairing under the condition
                sorted_indices = self.pairing(sc_a.detach())

                target_b = targets[sorted_indices]
                images_b = images[sorted_indices]
                sc_b = sc_a[sorted_indices]

                # element-wise normalization
                norm_sc_a = torch.div(sc_a, (sc_a+sc_b).detach())
                norm_sc_b = torch.div(
                    sc_b, (sc_a+sc_b).detach())
                with torch.no_grad():
                    if self.configs['label_mixing']=='saliency_label':
                        lam_a = norm_sc_a.mean(dim=[1, 2])
                        lam_b = norm_sc_b.mean(dim=[1, 2])
                    else:
                        raise NotImplementedError('--label_mixing \"{}\" is not implemented'.format(self.configs['label_mixing']))

                mask_a = torch.stack([norm_sc_a]*3, dim=1)
                mask_b = torch.stack([norm_sc_b]*3, dim=1)

                # Normalized MixUp
                combined_images = mask_a*images + mask_b*images_b
                loss, outputs = self._forward_all(combined_images,targets,target_b,lam_a,lam_b)
            else:
                loss, outputs = self._forward_all(images, targets)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            if self.configs['training_verbose']: # for efficiency
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(acc1[0].item()*100.0, images.size(0))
                top5.update(acc5[0].item()*100.0, images.size(0))

            # compute gradient and do SGD step
            if 'ap' not in self.configs['train_mode']:
                self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self._update_model(loss)
            self.step_lr_scheduler(epoch*len(loader))#+i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % int(len(loader)//2) == 0 and self.configs['local_rank']==0:
                progress.display(i)
            i += 1
        tok = time.time()        
        self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})

        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}

    def pairing_with_distance_mat(self, x:torch.tensor=None,bsz:int=0) -> torch.tensor:
        '''
        input:
            x: (batch_size, batch_size)
        '''
        if self.configs['condition']=='random':
            sorted_indices=np.random.permutation(bsz)
        elif self.configs['condition'] in ['greedy']:
            
            mask_value = 0
            sorted_indices = -1*np.ones(bsz, dtype=np.int32)
            
            np.fill_diagonal(x, mask_value)
            max_idx = np.argmax(x)
            row, col = max_idx//x.shape[0], max_idx % x.shape[0]
            first_row = copy.deepcopy(row)
            idx = 0
            sorted_indices[row] = col
            x[:, row] = mask_value
            while idx < x.shape[0] - 2:
                idx += 1
                row = col
                col = np.argmax(x[row])
                sorted_indices[row] = col
                x[:, row] = mask_value
            sorted_indices[col] = first_row
        elif self.configs['condition'] in ['greedy_c']:
            mask_value = -np.inf
            np.fill_diagonal(x, mask_value)
            sorted_indices = onecycle_cover(x)
        else:
            raise NotImplementedError('condition \"{}\" is not implemented'.format(self.configs['condition']))
        
        return sorted_indices


    def pairing(self, sc_a, sc_b=None):
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

        if self.configs['condition'] == 'greedy_c':
                X = distance_function(
                    sc_a, sc_b, self.configs['distance_metric']).cpu().numpy()
                sorted_indices=onecycle_cover(X)

        elif self.configs['condition'] != 'random':
            if self.configs['m_part'] >= self.configs['batch_size'] or sc_a.size(0) < self.configs['m_part']:
                X = distance_function(
                    sc_a, sc_b, self.configs['distance_metric']).cpu().numpy()
                target_indices = []
                sorted_indices = -1*np.ones(sc_a.size(0), dtype=int)
                if self.configs['condition'] == 'greedy':
                    max_idx = np.argmax(X)
                    row, col = max_idx//X.shape[0], max_idx % X.shape[0]
                    first_row = copy.deepcopy(row)
                    idx = 0
                    sorted_indices[row] = col
                    # print(row, col)
                    X[:, row] = 0
                    while idx < X.shape[0] - 2:
                        idx += 1
                        row = col
                        col = np.argmax(X[row])
                        sorted_indices[row] = col
                        X[:, row] = 0
                    sorted_indices[col] = first_row
                else:
                    raise NotImplementedError
            else:
                X = []
                split_sc = torch.split(
                    sc_a, self.configs['m_part'])
                for sp_sc in split_sc:
                    dist = distance_function(
                        sp_sc, sp_sc, self.configs['distance_metric']).cpu().numpy()
                    X.append(dist)
                greedy_indices = self.pool.starmap(
                    self.condition_func[self.configs['condition']], zip(X, self.start_idx))
                sorted_indices = []
                for g in greedy_indices:
                    sorted_indices += g.tolist()
        else:  # random
            sorted_indices = torch.randperm(sc_a.size(0), device=self.device)

        return sorted_indices
