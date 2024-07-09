import time
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import torch
import numpy as np
import torch.nn.functional as F



class MixupLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(MixupLearner, self).__init__(
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

            # images, target_a, target_b, lam_a, lam_b = mixup_data(
            #     images, targets, alpha=1)
            if self.configs['mix_prob']>=np.random.rand(1) and not finetune:
                lam = np.random.beta(self.configs['alpha'], self.configs['alpha'])# in the paper alpha mean beta in 0.2

                sorted_indices = torch.randperm(images.size(0), device=self.device)

                mixed_images = lam * images + (1 - lam) * images[sorted_indices, :]
                target_a, target_b = targets, targets[sorted_indices]
                lam_b=1.-lam
                lam_a=lam

                loss,outputs=self._forward_all(mixed_images, target_a, target_b, lam_a, lam_b)
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
        tok = time.time()        
        self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})

        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
