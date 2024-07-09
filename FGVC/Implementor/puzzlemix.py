import os, time
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import torch
import numpy as np
import torch.nn as nn
from Utils.puzzlemix_augmentation.puzzlemix import cost_matrix, mixup_graph
from multiprocessing import Pool

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes),
                           dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


class PuzzleMixLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(PuzzleMixLearner, self).__init__(
            model, file_name, save_path, device, configs)


        self.bce_loss = nn.BCELoss()  # .to(self.device)
        self.softmax = nn.Softmax(dim=1)  # .to(self.device)

        self.cost_matrix_dict = {'2': cost_matrix(2,self.configs['device']).unsqueeze(0), '4': cost_matrix(
            4,self.configs['device']).unsqueeze(0), '8': cost_matrix(8,self.configs['device']).unsqueeze(0), '16': cost_matrix(16,self.configs['device']).unsqueeze(0),
            '32': cost_matrix(32,self.configs['device']).unsqueeze(0),
            '64': cost_matrix(64,self.configs['device']).unsqueeze(0)}

        if self.configs['mp'] > 0:
            self.mp = Pool(self.configs['mp'])
        else:
            self.mp = None

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
        criterion_batch=self.criterion.__class__(reduction='none')
        try:
            for minibatch in loader:
                if len(minibatch)==2:
                    images, targets = minibatch
                else:
                    images, targets,_=minibatch
                self.optimizer.zero_grad()
                images, targets = images.to(self.device), targets.to(self.device)
                do_mixup = self.configs['mixup_epochs']<epoch and np.random.rand()<self.configs['mix_prob']
                if do_mixup and not finetune:

                    # process for Puzzle Mix
                    # random start
                    input_var = images.detach()
                    input_var.requires_grad = True

                    # calculate saliency (unary)
                    if self.configs['clean_lam'] == 0:
                        self.model.eval()
                        loss_clean, outputs = self._forward_all(input_var, targets)
                        self._just_backward(loss_clean)
                        self.model.train()
                        self.optimizer.zero_grad()

                    else:
                        # gradient regularization
                        loss_clean, outputs = self._forward_all(input_var, targets)
                        loss_clean *= self.configs['clean_lam']
                        self._just_backward(loss_clean, retain_graph=True)

                    unary = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1))
                    # perform mixup
                    alpha = np.random.beta(self.configs['alpha'], self.configs['alpha'])
                    rand_index = torch.randperm(images.size()[0]).to(self.device)
                    
                    block_num = 2 ** np.random.randint(1, 5)

                    with torch.no_grad():
                        combined_images, lam = mixup_graph(images, unary, rand_index,cost_matrix_dict=self.cost_matrix_dict,block_num=block_num,
                                                        alpha=alpha, beta=self.configs['puzzle_beta'], gamma=self.configs[
                                                            'puzzle_gamma'], eta=self.configs['puzzle_eta'],
                                                        neigh_size=self.configs['neigh_size'], n_labels=self.configs['n_labels'], transport=self.configs['transport'],
                                                        t_eps=self.configs['t_eps'], t_size=self.configs['t_size'],mp=self.mp,device=self.configs['device'])
                    loss, outputs = self._forward_all(combined_images, targets,targets[rand_index], lam, 1.-lam)
                else:
                    loss, outputs = self._forward_all(images, targets)
                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                if self.configs['training_verbose']: # for efficiency
                    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                    top1.update(acc1[0].item()*100.0, images.size(0))
                    top5.update(acc5[0].item()*100.0, images.size(0))
                # compute gradient and do SGD step
                self._update_model(loss)
                self.step_lr_scheduler(epoch*len(loader)+i)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % int(len(loader)//2) == 0 and self.configs['local_rank']==0:
                    progress.display(i)
                i += 1
            tok = time.time()        
            self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})


            return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
        except KeyboardInterrupt:
            self.logger.info('[train] Interrupt')
            self.mp.close()
            raise KeyboardInterrupt
