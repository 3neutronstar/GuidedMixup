import os, copy, time
import torch
import logging
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import wandb
from Utils.label_smoothing_loss import mixup_target
from Utils.calc_score import AverageMeter, ProgressMeter, accuracy
from Utils.label_smoothing_loss import LabelSmoothingLoss
from Utils.logger import convert_secs2time, set_logging_defaults, create_distributed_logging
from Utils.custom_crossentropy import CategoricalBCE, OnehotCrossEntropyLoss
from Utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from Utils.warmup_scheduler import WarmupConstantSchedule # for mixed precision
from torch.cuda.amp import autocast
from Utils.logger import reduce_value

try:
    from pairing import onecycle_cover
except: pass

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except: pass


class Baseline:
    def __init__(self, model, file_name, save_path, device, configs):
        self.file_name = file_name
        self.save_path = save_path
        self.device = device
        self.configs = configs
        # dataset normalize factor
        self.mean=(configs['mean'].squeeze()).cpu().tolist()
        self.std=(configs['std'].squeeze()).cpu().tolist()

        ## logger ##
        if not self.configs['ddp'] and self.configs['local_rank']==0:
            set_logging_defaults(save_path,file_name)
            self.logger = logging.getLogger('main')
            self.summaryWriter = SummaryWriter(os.path.join(
                self.save_path, file_name))
        elif self.configs['ddp'] and self.configs['local_rank']==0: # distributed ddp
            if int(torch.__version__.split('.')[0])>=2 :
                #torch 2.0 with torchrun
                set_logging_defaults(save_path,file_name)
                self.logger = logging.getLogger('main')
                self.summaryWriter = SummaryWriter(os.path.join(
                    self.save_path, file_name))
            else:
                #torch 1.x with distributed
                self.logger = create_distributed_logging(save_path,file_name)
                self.summaryWriter = SummaryWriter(os.path.join(
                    self.save_path, file_name))
        self.best_valid_accuracy = 0.0
        self.best_valid_loss=0.0
        self.best_valid_top5=0.0
        ############

        ## Hyper Parameter setting ##
        if self.configs['criterion']=='bce':
            self.criterion=CategoricalBCE().to(self.device)        
            self.none_reduction_criterion=CategoricalBCE(reduction='none').to(self.device)        
        elif self.configs['criterion']=='label_smoothing': # 
            self.criterion = LabelSmoothingLoss(self.configs['num_classes'],smoothing=self.configs['smoothing']).to(self.device)
            self.none_reduction_criterion = LabelSmoothingLoss(self.configs['num_classes'],smoothing=self.configs['smoothing'],reduction='none').to(self.device)
        elif self.configs['criterion']=='crossentropy': # cross entropy
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.none_reduction_criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        elif self.configs['criterion']=='onehot_ce': # cross entropy
            self.criterion = OnehotCrossEntropyLoss() #nn.BCELoss().to(self.device)
            self.none_reduction_criterion = OnehotCrossEntropyLoss(reduction='none') #nn.BCELoss().to(self.device)
        else:
            raise NotImplementedError
        self.bce_criterion=nn.BCELoss().to(self.device)
        self.onehotcrossentropy = OnehotCrossEntropyLoss().to(self.device) #nn.BCELoss().to(self.device)

        # model setup
        self.model = model
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if self.configs['local_rank']==0:
            print('Total parameter number:', params, '\n')
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_all_mb = (param_size + buffer_size) / 1024**2
            print('model size: {:.3f}MB'.format(size_all_mb))

        if self.configs['wandb'] and self.configs['wandb_gradient_tracking']:
            wandb.watch(self.model,log='gradients',log_freq=100)
        
        self.img_save_root=os.path.join(self.save_path, self.file_name, 'images')
        if not os.path.exists(self.img_save_root):
            os.makedirs(self.img_save_root, exist_ok=True)

    def _save_images(self, images, name, normalize=True):
        torchvision.utils.save_image(
            images, os.path.join(self.save_path, name),nrow=images.size(0),normalize=normalize)

    def step_lr_scheduler(self,epoch):
        if epoch <= (self.configs['epochs']-self.configs['cooldown']+1)*self.train_steps and self.configs['lr_scheduler'] in ['cosine','step']:
            if self.configs['warmup']>0 and epoch<self.configs['warmup']*self.train_steps and self.configs['lr_scheduler'] in ['cosine','step']:
                self.warmup_lr_scheduler.step(epoch)
            else:
                self.lr_scheduler.step(epoch)
        else:
            pass
    
    def _before_train(self,train_loader, valid_loader):
        #################### optimizer
        if self.configs['mode']=='train':
            if self.configs['optimizer']=='sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(
                ), self.configs['lr'], self.configs['momentum'], weight_decay=self.configs['weight_decay'], nesterov=self.configs['nesterov'])
            elif self.configs['optimizer']=='adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(
                ), self.configs['lr'], weight_decay=self.configs['weight_decay'])
        else:
            raise NotImplementedError

        ############### lr scheduling
        if self.configs['lr_scheduler']=='step':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, len(train_loader)*np.array(self.configs['lr_steps']), self.configs['gamma'])
        elif self.configs['lr_scheduler']=='cosine':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, len(train_loader)*(1+self.configs['epochs']), eta_min=self.configs['lr_min'])
        elif self.configs['lr_scheduler']=='cosine_warmup':
            self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0= len(train_loader)*self.configs['epochs'],T_mult=1,T_up=len(train_loader)*self.configs['warmup'], eta_max=self.configs['lr'])
        else:
            raise NotImplementedError
        self.train_steps=len(train_loader)

        if self.configs['warmup']>0 and self.configs['lr_scheduler'] in ['cosine','sgd']:
            self.warmup_lr_scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps=len(train_loader)*self.configs['warmup'], last_epoch=self.configs['epochs']-self.configs['cooldown'])
        ##################
        
        # apex        
        if not self.configs['apex'] and self.configs['use_fp16']:
            # if use_fp16 false: scaler is not used
            # else: scaler is used
            if self.configs['mixup_type']=='hidden':
                init_scale=8192
            else:
                init_scale=2**16
            self.scaler=torch.cuda.amp.GradScaler(init_scale=init_scale,enabled=self.configs['use_fp16'])
            self.use_amp=True
        else:
            self.use_amp=False

        if self.configs['apex']:
            self.model.to(self.configs['device'])
            if self.configs['use_fp16']:
                opt_level='O1'
            else: opt_level='O0'
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)
            if self.configs['ddp']:
                self.model=DDP(self.model)
            # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        else:
            # ddp
            if self.configs['ddp']:
                if int(torch.__version__.split('.')[0])>=2: 
                    # pytorch 2.x with torchrun
                    self.model.cuda(self.configs['local_rank'])
                    self.model = DistributedDataParallel(self.model, device_ids=[self.configs['local_rank']], output_device = self.configs['local_rank'])
                else: 
                    # pytorch 1.x with torch.distributed.launch
                    # self.scaler = GradScaler() # for mixed precision
                    self.model.cuda(self.configs['local_rank'])
                    self.model = DistributedDataParallel(self.model, device_ids=[self.configs['local_rank']])
            else:
                self.model = nn.DataParallel(self.model).to(self.configs['device'])
        
        if self.configs['compile']==True:
            if int(torch.__version__.split('.')[0])>=2 and torch.cuda.get_device_capability()[0] >= 7: # acceleration by PyTorch 2.0
                try:
                    #torch._dynamo.config.verbose=True 
                    torch._dynamo.config.suppress_errors = True
                    model=torch.compile(model)
                    if self.configs['local_rank']==0:
                        print("Torch Compile")
                except:
                    if self.configs['local_rank']==0:
                        print("Compile try fails")
            else:
                if self.configs['local_rank']==0:
                    print("This system does not satisfy compiling option's requirements, use normal --compile False")
        else:
            if self.configs['local_rank']==0:
                print("Not using model.compile()")


    def run(self, train_loader, valid_loader):
        self._before_train(train_loader,valid_loader)

        ########################
        ####### training #######
        ########################
        tik = time.time()
        self.learning_time=AverageMeter('Time', ':6.3f')
        for epoch in range(1, self.configs['epochs'] + 1):
            if self.configs['ddp']:
                train_loader.sampler.set_epoch(epoch)
            epoch_tik=time.time()

            train_info = self._train(train_loader, epoch)
            if self.configs['ddp']:
                torch.distributed.barrier()
            valid_info = self._eval(valid_loader, epoch)
            if self.configs['local_rank']==0:
                for key in train_info.keys():
                    self.summaryWriter.add_scalar(f'train/{key}',train_info[key],epoch)
                for key in valid_info.keys():
                    self.summaryWriter.add_scalar(f'eval/{key}',valid_info[key],epoch)
                self.summaryWriter.flush()        ## save best model ##
                if self.best_valid_accuracy < valid_info['accuracy']:
                    self.best_valid_accuracy = valid_info['accuracy']
                    self.best_valid_loss=valid_info['loss']
                    self.best_valid_top5=valid_info['top5']
                    if self.configs['apex']:
                        model_dict = self.model.state_dict()
                    else:
                        model_dict = self.model.module.state_dict()
                    optimizer_dict = self.optimizer.state_dict()
                    save_dict = {
                        'info': valid_info,
                        'model': model_dict,
                        'optim': optimizer_dict,
                    }
                    torch.save(save_dict, os.path.join(
                        self.save_path, self.file_name, 'best_model.pt'))
                    print("Save Best Accuracy Model")
            if self.configs['ddp']:
                torch.distributed.barrier()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            self.learning_time.update(time.time()-epoch_tik)
        tok = time.time()
        print("Learning Time: {:.4f}s".format(tok-tik))
        ############################
        ####### end training #######
        ############################

        ###############################
        ####### Final logging #########
        ###############################
        if self.configs['local_rank'] ==0:
            df_dict=copy.deepcopy(self.configs)
            df_dict.update({'learning_time':tok-tik,
            'time':self.file_name,
            'valid_loss':self.best_valid_loss,
            'valid_acc':self.best_valid_accuracy,
            'train_loss':train_info['loss'],
            'train_acc':train_info['accuracy'],
        })
        if self.configs['wandb'] and self.configs['local_rank']==0:
            wandb.config.update(df_dict,allow_val_change=True)

        ## Evaluation ##
        if self.configs['train_mode']=='comixup' and self.configs['mixup_parallel']:
            self.mpp.close()
        ###############
        if self.configs['local_rank']==0:
            for key in df_dict.keys():
                if isinstance(df_dict[key], torch.Tensor):
                    df_dict[key]=df_dict[key].view(-1).detach().cpu().tolist()
                if type(df_dict[key])==list:
                    df_dict[key]=','.join(str(e) for e in df_dict[key])
                df_dict[key]=[df_dict[key]]
            if self.configs['wandb']:
                wandb.config.update(df_dict,allow_val_change=True)
            df_cat=pd.DataFrame.from_dict(df_dict,dtype=object)
            if os.path.exists('./learning_result.csv'):
                df=pd.read_csv('./learning_result.csv',index_col=0,dtype=object)
                
                df=pd.merge(df,df_cat,how='outer')
            else: df=df_cat
            df.to_csv('./learning_result.csv')
            self.logger.info("[Best Valid Accuracy] {:.2f} [Saved Time] {}".format(
                self.best_valid_accuracy,self.file_name))
            ##############
        return

    def _forward_all(self,images, targets, target_b=None,lam_a=1.0,lam_b=0.0, rand_index=None,extract_feature=False):
        # print("HI")
        target_dim=targets.dim()
        assert target_dim in [1,2], "target dim should be 1 or 2, but {}".format(target_dim)
        # forward model and criterion
        if rand_index is not None and target_b is None:
            target_b=targets[rand_index]
            if lam_b == 0.0:
                lam_b=1.0-lam_a

        cast_dict=dict(enabled=self.use_amp)
        # torch version > 2.0 then use dtype for autocast
        if torch.__version__>'1.9.0':
            cast_dict.update({'dtype':torch.float16})

        with autocast(**cast_dict):
            outputs=self.model(images)
            outputs=self.model(images,extract_feature=extract_feature)
            if extract_feature:
                outputs,features=outputs
            if targets.dim()==2: # already mixed
                if self.configs['criterion']=='bce':
                    loss = self.criterion(outputs, targets)
                else:
                    loss = self.onehotcrossentropy(outputs, targets)
            else:
                assert targets.dim()==1, "targets dim should be 1, but {}".format(targets.dim())
                if target_b is None: # original propgation
                    if self.configs['criterion'] == 'label_smoothing':
                        targets=mixup_target(targets, num_classes=self.configs['num_classes'],lam=1, lam_b=0., smoothing=self.configs['smoothing'], device=self.device)
                        loss=self.onehotcrossentropy(outputs,targets)
                    else:
                        loss = self.criterion(outputs, targets)
                else:
                    # if type(lam_a)==torch.Tensor:
                    if self.configs['criterion']=='bce':
                        mixed_targets = mixup_target(targets, num_classes=self.configs['num_classes'],lam=lam_a, lam_b=lam_b, smoothing=0.0, device=self.device,target_b=target_b)
                        loss = self.criterion(outputs, mixed_targets)
                    elif self.configs['criterion'] == 'label_smoothing':
                        mixed_targets = mixup_target(targets, num_classes=self.configs['num_classes'],lam=lam_a, lam_b=lam_b, smoothing=self.configs['smoothing'], device=self.device,target_b=target_b)
                        loss=self.onehotcrossentropy(outputs,mixed_targets)
                    elif self.configs['criterion'] in ['crossentropy']:
                        if isinstance(lam_a, torch.Tensor):
                            loss = (self.none_reduction_criterion(outputs, targets)*lam_a+self.none_reduction_criterion(outputs, target_b)*lam_b).mean()
                        elif isinstance(lam_a, float):
                            loss = self.criterion(outputs, targets)*lam_a+self.criterion(outputs, target_b)*lam_b
                        else:
                            raise NotImplementedError('lam should be either float or torch.Tensor. Type: {}'.format(type(lam_a)))
                    else:
                        raise NotImplementedError('criterion {} is not implemented'.format(self.configs['criterion']))
        if extract_feature:
            return loss, (outputs,features)
        return loss, outputs
 
    def _update_model(self, loss,retain_graph=False):
        if self.configs['apex']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
            self.optimizer.step()
        else:
            if self.configs['use_fp16']:
                cast_dict=dict(enabled=self.use_amp)
                # torch version > 2.0 then use dtype for autocast
                if torch.__version__>'1.9.0':
                    cast_dict.update({'dtype':torch.float16})
                with autocast(cast_dict):
                    self.scaler.scale(loss).backward(retain_graph=retain_graph)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward(retain_graph=retain_graph)
                self.optimizer.step()
        return
    
    def _just_backward(self, loss, retain_graph=False):
        if self.configs['apex']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        else:
            if self.configs['use_fp16']:
                self.scaler.scale(loss).backward(retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
    
    def _train(self, loader, epoch, finetune=False):
        tik = time.time()
        self.model.train()
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
        i = 0
        end = time.time()
        for minibatch in loader:
            if len(minibatch)==2:
                images, target = minibatch
            else:
                images, target,_=minibatch
            # measure data loading time
            images, target = images.to(
                self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # compute output
            loss,output=self._forward_all(images, target)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            if self.configs['training_verbose']: # for efficiency
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
    
    def _show_training_log(self,epoch,log_dict:dict):
        if self.configs['local_rank']==0:
            if self.configs['training_verbose']: # for efficiency
                self.logger.info('[train] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
                    log_dict['loss'], log_dict['accuracy'], log_dict['top5'], log_dict['time']))
                if self.configs['wandb']:
                    wandb.log(
                    {'train_loss': log_dict['loss'], 'train_top1': log_dict['accuracy'], 'train_top5': log_dict['top5']}, step=epoch)
            else:
                self.logger.info('[train] Lr: {:.4e} | Loss: {:.4f} | time: {:.3f}'.format(
                    self.optimizer.param_groups[0]['lr'], log_dict['loss'], log_dict['time']))
                if self.configs['wandb']:
                    wandb.log(
                    {'train_loss': log_dict['loss']}, step=epoch)

    def _eval(self, loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.eval()
        end = time.time()
        # optimizer zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        i = 0
        with torch.no_grad():
            for images, target in loader:

                # measure data loading time
                images, target = images.to(
                    self.device), target.to(self.device)

                # compute output
                output = self.model(images)
                if isinstance(output,tuple):
                    output = output[0]
                loss = F.cross_entropy(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(reduce_value(loss).item(), images.size(0))
                top1.update(reduce_value(acc1[0]*100.0).item(), images.size(0))
                top5.update(reduce_value(acc5[0]*100.0).item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                i += 1

        log_dict={'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}
        if self.configs['local_rank']==0:
            self._show_eval_log(epoch,log_dict)
        #####################
        return log_dict
    
    def _show_eval_log(self,epoch,log_dict):
        hour,minute,second=convert_secs2time(self.learning_time.avg*(self.configs['epochs']-epoch))
        self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} [Time Left] {:2d}h {:2d}m {:2d}s'.format(epoch,
            log_dict['loss'], log_dict['accuracy'], log_dict['top5'], hour,minute,second))
        if self.configs['wandb']:
            wandb.log({'eval_loss': log_dict['loss'], 'eval_top1':  log_dict['accuracy'], 'eval_top5': log_dict['top5'],'eval_best_top1':self.best_valid_accuracy,'eval_best_top5':self.best_valid_top5},step=epoch)


    def load_model(self, model, name=None):
        if self.file_name is not None:
            if name == None:
                model_dict = torch.load(os.path.join(
                    self.save_path, self.file_name, 'best_model.pt'),map_location=self.device)
            else:
                model_dict = torch.load(os.path.join(
                    self.save_path, name, 'best_model.pt'),map_location=self.device)
            print("Model Performance: ", model_dict['info'])
            model.load_state_dict(model_dict['model'],strict=False)
            print("Load Best Model Complete")

    def evaluation(self, loader):
        self.model.eval()
        # load best model
        self.load_model(self.model.module)
        
        i=0
        losses = AverageMeter('NLL Loss', ':.4e')
        accs=AverageMeter('Accuracy', ':6.2f')
        for minibatch in loader:
            if len(minibatch)==2:
                images,target=minibatch
            elif len(minibatch)==3:
                images,target,_=minibatch
            # measure data loading time
            images, target = images.to(
                self.device), target.to(self.device)

            # compute output
            output = self.model(images)
            if isinstance(output,tuple):
                output = output[0]

            loss = F.cross_entropy(output, target)
            losses.update(loss.item(), images.size(0))

            # get acc
            acc = accuracy(output, target, topk=(1, 5))
            accs.update(acc[0].item(), images.size(0))
            # measure elapsed time
            i += 1
            
        return {'acc':accs.avg,'loss':losses.avg}