import time
from Implementor.baseline import Baseline
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
from Utils.comixup_augmentation.mixup_parallel import MixupProcessParallel
from Utils.comixup_augmentation.mixup import mixup_process
import torch
import torch.nn.functional as F
import torch.nn as nn

from Utils.custom_crossentropy import OnehotCrossEntropyLoss


def to_one_hot(inp, num_classes, device='cuda'):
    y_onehot = torch.zeros((inp.size(0), num_classes),
                           dtype=torch.float32,
                           device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)

    return y_onehot


def distance(z, dist_type='l2'):
    '''Return distance matrix between vectors'''
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist


class CoMixupLearner(Baseline):
    def __init__(self, model, file_name, save_path, device, configs):
        super(CoMixupLearner, self).__init__(
            model, file_name, save_path, device, configs)

        self.criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()

        self.onehotcrossentropy = OnehotCrossEntropyLoss()  # nn.BCELoss().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        if self.configs['mixup_parallel']:
            self.configs.pop('mean')
            self.configs.pop('std')
            torch.multiprocessing.set_start_method('spawn')
            gpu_ids = self.configs['gpu_ids'].split(",")
            self.mpp = MixupProcessParallel(
                self.configs['m_part'], self.configs['batch_size'], gpu_ids)
            
        if self.configs['criterion']=='label_smoothing':
            raise NotImplementedError('label smoothing is not supported for comixup')

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
        times=[]
        for minibatch in loader:
            if len(minibatch)==2:
                images, targets = minibatch
            else:
                images, targets,_=minibatch
            images, targets = images.to(self.device), targets.to(self.device)
            if self.configs['mixup_epochs'] < epoch and not finetune:

                input_var = images.detach().clone()
                input_var.requires_grad = True
                target_var = targets
                A_dist = None
                sc = None
                self.optimizer.zero_grad()

                # Calculate saliency (unary)
                if self.configs['clean_lam'] == 0:
                    self.model.eval()
                    loss_batch, outputs=self._forward_all(input_var, target_var)
                else:
                    loss_batch, outputs=self._forward_all(input_var, target_var)
                    loss_batch*=self.configs['clean_lam']
                self._just_backward(loss_batch, retain_graph=True)

                sc = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

                # Here, we calculate distance between most salient location (Compatibility)
                # We can try various measurements
                with torch.no_grad():
                    z = F.avg_pool2d(sc, kernel_size=8, stride=1)
                    z_reshape = z.reshape(z.size(0), -1)
                    # z_reshape = z.reshape(self.configs['batch_size'], -1)
                    z_idx_1d = torch.argmax(z_reshape, dim=1)
                    z_idx_2d = torch.zeros(
                        (z.size(0), 2), device=z.device)
                    # (self.configs['batch_size'], 2), device=z.device)
                    z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
                    z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
                    A_dist = distance(z_idx_2d, dist_type='l1')

                if self.configs['clean_lam'] == 0:
                    self.model.train()
                    self.optimizer.zero_grad()

                # Perform mixup and calculate loss
                target_reweighted = to_one_hot(
                    targets, self.configs['num_classes'])
                aug_tik=time.time()
                if self.configs['mixup_parallel']:
                    with torch.no_grad():
                        out, target_reweighted = self.mpp(images.cpu(),
                                                    target_reweighted.cpu(),
                                                    configs=self.configs,
                                                    sc=sc.cpu(),
                                                    A_dist=A_dist.cpu())
                    out = out.to(self.device)
                    target_reweighted = target_reweighted.to(self.device)

                else:
                    out, target_reweighted = mixup_process(images,
                                                        target_reweighted,
                                                        configs=self.configs,
                                                        sc=sc,
                                                        A_dist=A_dist)
                aug_tok=time.time()
                times.append(aug_tok-aug_tik)
                loss,outputs=self._forward_all(out, target_reweighted)
            else:
                loss,outputs=self._forward_all(images,targets)

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
                #print(torch.tensor(times).mean())
            i += 1
        tok = time.time()
        self._show_training_log(epoch,{'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'time': tok-tik})

        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
