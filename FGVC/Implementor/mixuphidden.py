from multiprocessing import Pool
from Implementor.baseline import Baseline
from Model.layer.dropblock import DropBlock
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Utils.label_smoothing_loss import mixup_target
import math

class MixupHiddenLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(MixupHiddenLearner, self).__init__(
            model, file_name, save_path, device, configs)

        if self.configs['mixup_type'] not in ['hidden']:
            raise NotImplementedError(
                'MixupHiddenLearner only support hidden mixup: Describe the mixup_type as hidden or hook_hidden')

        self.configs.pop('mean')
        self.configs.pop('std')


    # for manual alpha scehduling
    def cosine_annealing_scheduler(self, iteration, total_iterations, start_value = 0.5, end_value = 10):
        """Cosine annealing scheduler for smoothly increasing values."""
        fraction = iteration / float(total_iterations - 1)
        cosine_annealed = (1 + math.cos(fraction * math.pi)) / 2
        return end_value + (start_value - end_value) * cosine_annealed
    
    def _init_train(self, epoch):
        self.configs['current_epoch'] = epoch
            
        # prepare for parallel mixup
        if self.configs['condition'] not in ['greedy_c', 'random'] and self.configs['batch_size'] > self.configs['m_part']:
            if self.configs['m_part'] < self.configs['batch_size']:
                self.n_proc = self.configs['batch_size']//self.configs['m_part']
                self.configs['start_idx'] = [i*self.configs['m_part']
                                             for i in range(self.n_proc)]
                self.configs['end_idx'] = []
                error_count = 0
                for i in range(self.n_proc):
                    if (i+1)*self.configs['m_part'] <= self.configs['batch_size']:
                        self.configs['end_idx'].append(
                            (i+1)*self.configs['m_part'])
                    else:
                        self.configs['end_idx'].append(
                            self.configs['batch_size'])
                        if error_count == 1:
                            raise NotImplementedError
                        error_count += 1
                if self.configs['condition'] == 'bruteforce_pair':
                    self.configs['start_idx'].append(
                        self.configs['m_part']*self.n_proc)
                else:
                    self.configs['mp'] = Pool(self.n_proc)
        elif self.configs['condition'] in ['greedy']:
            self.configs['mp'] = None
            self.configs['start_idx'] = None
            self.configs['end_idx'] = None
        else:
            self.configs['start_idx'] = None
            self.configs['end_idx'] = None
            self.configs['mp'] = None
            # self.configs['m_part'] = None

        
    def _end_train(self, epoch=None):
        epoch=self.configs.pop('current_epoch')
        # prepare for parallel mixup
        if self.configs['condition'] not in ['random', 'greedy_c'] and self.configs['batch_size'] > self.configs['m_part']:
            self.configs['mp'].close()
            self.configs.pop('mp')

############# basic train function using mixup ################
    def _train(self, loader, epoch, finetune=False):
        self._init_train(epoch)
        tik = time.time()
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
            get_mixed_feature=False
        i = 0
        self.model.train()
        end = time.time()

        if self.configs['mix_prob_scheduling']:
            self.configs['mix_prob'] = 1.*epoch/self.configs['epochs']
        for minibatch in loader:
            #### prepare data ####
            if len(minibatch) == 2:
                images, targets = minibatch
            else:
                images, targets, _ = minibatch
            # measure data loading time
            images, targets = images.to(
                self.device), targets.to(self.device)

            # determine whether to do mixup
            # do mixup or not
            mixup = self.configs['mixup_epochs'] <= epoch and self.configs['mix_prob'] >= np.random.rand()
            # mixup
            if mixup and not finetune:
                #### extract saliency if needed ####
                if 'guided' not in self.configs['train_mode'] and self.configs['condition'] != 'random':
                    sorted_indices = self.pairing_sr(images, targets)
                    sc = None
                else:
                    sorted_indices = None
                    sc = None

                # compute outputs
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs, targets, targets_b, lam_a, lam_b, verboses = self.model(
                        images, targets, mixup=True, sc=sc, mixup_indices=sorted_indices, get_mixed_feature=get_mixed_feature,configs=self.configs)  # images, targets, mixup, configs
                
                ################################ loss ###########################
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.configs['criterion']=='bce':
                        mixed_targets = mixup_target(targets, self.configs['num_classes'], sorted_indices,lam_a,lam_b, 0.0, self.device)
                        loss = self.criterion(outputs, mixed_targets)
                    elif self.configs['criterion'] == 'label_smoothing':
                        mixed_targets = mixup_target(targets, self.configs['num_classes'], sorted_indices,lam_a, lam_b, self.configs['smoothing'], self.device)
                        loss=self.onehotcrossentropy(outputs,mixed_targets)
                    elif self.configs['criterion'] in ['crossentropy']:
                        if isinstance(lam_a, torch.Tensor):
                            loss = (self.none_reduction_criterion(outputs, targets)*lam_a+self.none_reduction_criterion(outputs, targets_b)*lam_b).mean()
                        elif isinstance(lam_a, float):
                            loss = self.criterion(outputs, targets)*lam_a+self.criterion(outputs, targets_b)*lam_b
                        else:
                            raise NotImplementedError('lam should be either float or torch.Tensor. Type: {}'.format(type(lam_a)))
                    else:
                        raise NotImplementedError('criterion {} is not implemented'.format(self.configs['criterion']))
                ###############################################################
            else: # normal training
                loss, outputs=self._forward_all(images, targets)


            if not self.configs['train_mode'].endswith('-ap') or (self.configs['train_mode'].endswith('-ap') and self.configs['mixup_epochs'] > epoch):
                self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self._update_model(loss)
            self.step_lr_scheduler(epoch*len(loader)+i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            if self.configs['training_verbose'] and self.configs['local_rank']==0:  # for efficiency
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(acc1[0].item()*100.0, images.size(0))
                # compute gradient and do SGD step
                top5.update(acc5[0].item()*100.0, images.size(0))

            if i % int(len(loader)//2) == 0 and self.configs['local_rank'] == 0:
                progress.display(i)
            i += 1
        tok = time.time()
        log_dict={'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
        show_log_dict=log_dict.copy()
        show_log_dict.update({'time': tok-tik})
        self._show_training_log(epoch, show_log_dict)
        self._end_train()

        return log_dict
    