import time
import torch
import math
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
from Mixup.utils import rand_bbox
import numpy as np
import torch.nn.functional as F

from Utils.label_smoothing_loss import mixup_target

class CutMixLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(CutMixLearner, self).__init__(
            model, file_name, save_path, device, configs)


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
                lam = np.random.beta(
                    self.configs['alpha'], self.configs['alpha'])
                sorted_indices = torch.randperm(images.size(0), device=self.device)
                target_a = targets
                target_b = targets[sorted_indices]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[sorted_indices,
                                                            :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()
                           [-1] * images.size()[-2]))  # transformed ratio

                # Official code
                lam_b=1.-lam
                #lam_b=math.sqrt(1.-lam**2+1e-12)
                loss,outputs=self._forward_all(images, target_a, target_b, lam, lam_b)
            else:
                loss,outputs=self._forward_all(images, targets)

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
