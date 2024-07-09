from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
import numpy as np
import time
import torch
import torch.nn.functional as F
from Mixup.utils import get_spm, rand_bbox


class SnapMixLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(SnapMixLearner, self).__init__(
            model, file_name, save_path, device, configs)
        if 'resnet' in self.configs['model'] or 'densenet' in self.configs['model']:
            pass
        else:
            raise NotImplementedError  # snapmix only available in resnet and densenet

    def _train(self, loader, epoch, finetune=False):
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

        none_reduction_criterion = self.criterion.__class__(reduction='none')
        i = 0
        
        for minibatch in loader:
            if len(minibatch)==2:
                images, targets = minibatch
            else:
                images, targets,_=minibatch
            images, targets = images.to(self.device), targets.to(self.device)

            lam_a = torch.ones(images.size(0)).to(self.device)
            lam_b = 1 - lam_a
            targets_b = targets.clone()
            r = np.random.rand(1)
            if r < self.configs['mix_prob'] and not finetune:
                image_size = (images.size(2), images.size(3))
                wfmaps, _ = get_spm(images, targets, image_size, self.model,self.configs['model'])
                bs = images.size(0)
                rand_index = torch.randperm(bs).to(self.device)
                lam = np.random.beta(self.configs['alpha'], self.configs['alpha'])
                lam1 = np.random.beta(self.configs['alpha'], self.configs['alpha'])
                wfmaps_b = wfmaps[rand_index, :, :]
                targets_b = targets[rand_index]

                same_label = targets == targets_b
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(images.size(), lam1)

                area = (bby2-bby1)*(bbx2-bbx1)
                area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

                if area1 > 0 and area > 0:
                    ncont = images[rand_index, :,
                                bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
                    ncont = F.interpolate(ncont, size=(
                        bbx2-bbx1, bby2-bby1), mode='bilinear', align_corners=True)
                    images[:, :, bbx1:bbx2, bby1:bby2] = ncont
                    lam_a = 1 - wfmaps[:, bbx1:bbx2,
                                    bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
                    lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(
                        2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
                    tmp = lam_a.clone()
                    lam_a[same_label] += lam_b[same_label]
                    lam_b[same_label] += tmp[same_label]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                            (images.size()[-1] * images.size()[-2]))
                    lam_a[torch.isnan(lam_a)] = lam
                    lam_b[torch.isnan(lam_b)] = 1-lam
                    
            # outputs, _ = self.model(images)
            loss, outputs = self._forward_all(images, targets, targets_b, lam_a, lam_b, targets_b)
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
            i += 1
        tok = time.time()
        self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})
        
        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
