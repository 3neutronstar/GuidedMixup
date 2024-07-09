from Mixup.utils import saliency_bbox, saliency_bbox
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import numpy as np
import time
import torch
from Utils.custom_crossentropy import OnehotCrossEntropyLoss
import torch.nn.functional as F

from Utils.label_smoothing_loss import mixup_target

class SaliencyMixLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(SaliencyMixLearner, self).__init__(
            model, file_name, save_path, device, configs)
        self.custom_crossentropy = OnehotCrossEntropyLoss()

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
        i=0

        for minibatch in loader:
            if len(minibatch)==2:
                images, targets = minibatch
            else:
                images, targets,_=minibatch
            images, targets = images.to(self.device), targets.to(self.device)
            r = np.random.rand(1)
            if self.configs['alpha'] > 0 and r < self.configs['mix_prob'] and not finetune:
                # generate mixed sample
                lam = np.random.beta(
                    self.configs['alpha'], self.configs['alpha'])
                rand_index = torch.randperm(images.size()[0]).to(self.device)
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = saliency_bbox(
                    images[rand_index[0]], lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index,
                                                            :, bbx1:bbx2, bby1:bby2]
                lam_a = 1. - ((bbx2 - bbx1) * (bby2 - bby1) /
                                (images.size()[-1] * images.size()[-2]))
                lam_b = 1. - lam_a
                # compute output
                input_var = torch.autograd.Variable(images, requires_grad=True)
                target_a_var = torch.autograd.Variable(target_a)
                outputs = self.model(input_var)
                # # Official code
                if self.configs['criterion']=='bce':
                    mixed_targets = mixup_target(target_a_var, self.configs['num_classes'], rand_index,lam_a,lam_b, 0.0, self.device)
                    loss = self.criterion(outputs, mixed_targets)
                elif self.configs['criterion'] == 'label_smoothing':
                    mixed_targets = mixup_target(target_a_var, self.configs['num_classes'], rand_index,lam_a, lam_b, self.configs['smoothing'], self.device)
                    loss=self.onehotcrossentropy(outputs,mixed_targets)
                elif self.configs['criterion'] in ['crossentropy']:
                    if isinstance(lam_a, torch.Tensor):
                        loss = (self.none_reduction_criterion(outputs, target_a_var)*lam_a+self.none_reduction_criterion(outputs, target_b)*lam_b).mean()
                    elif isinstance(lam_a, float):
                        loss = self.criterion(outputs, target_a_var)*lam_a+self.criterion(outputs, target_b)*lam_b
                    else:
                        raise NotImplementedError('lam should be either float or torch.Tensor. Type: {}'.format(type(lam_a)))
                else:
                    raise NotImplementedError('criterion {} is not implemented'.format(self.configs['criterion']))

            else:
                loss,outputs=self._forward_all(images,targets)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            if self.configs['training_verbose']: # for efficiency
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(acc1[0].item()*100.0, images.size(0))
                top5.update(acc5[0].item()*100.0, images.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            self._update_model(loss)
            self.step_lr_scheduler(epoch*len(loader)+i)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % int(len(loader)//2) == 0 and self.configs['local_rank']==0:
                progress.display(i)
            i+=1
        tok=time.time()
        self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})
        
        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
