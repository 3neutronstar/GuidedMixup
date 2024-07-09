import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.label_smoothing_loss import LabelSmoothingLoss
class OnehotCrossEntropyLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(OnehotCrossEntropyLoss,self).__init__()
        self.reduction=reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,prediction,target):
        if self.reduction=='mean':
            return torch.mean(torch.sum(-target * self.logsoftmax(prediction), dim=1))
        elif self.reduction=='sum':
            return torch.sum(torch.sum(-target * self.logsoftmax(prediction), dim=1))
        else:
            return torch.sum(-target * self.logsoftmax(prediction), dim=1)


class CategoricalBCE(nn.Module):
    '''
        BCE(softmax(logit) , onehottarget)
        input: numclasses = default: None (none is automatically inferred)
        input2: reduction = default:'mean' 
    '''
    def __init__(self,num_classes=None,reduction='mean'):
        super(CategoricalBCE,self).__init__()
        self.reduction=reduction
        self.bce=nn.BCELoss(reduction=reduction)
        self.num_classes=num_classes

    
    def forward(self,prediction,target):

        softened_logits=torch.softmax(prediction,dim=1)
        if target.shape!=prediction.shape:
            if self.num_classes is None:
                num_classes = prediction.size(1)
            else:
                num_classes = self.num_classes
            onehot_target=F.one_hot(target,num_classes).float()
        else:
            onehot_target=target
        # print(softened_logits.shape,onehot_target.shape)
        if self.reduction == 'none':
            return self.bce(softened_logits,onehot_target).mean(dim=1)
        else:
            return self.bce(softened_logits,onehot_target)#.mean(dim=1) # aware! it use mean


class MixupClassificationLoss(nn.Module):
    def __init__(self, criterion_type='crossentropy',num_classes=None,reduction='mean',smoothing=0.0):
        self.criterion_type=criterion_type
        self.num_classes=num_classes
        self.reduction=reduction
        if self.criterion_type == 'crossentropy':
            self.criterion=nn.CrossEntropyLoss(reduction=self.reduction)
        elif self.criterion_type == 'bce':
            self.criterion=CategoricalBCE(num_classes=self.num_classes,reduction=self.reduction)
        elif self.criterion_type == 'onehot_ce':
            self.criterion=OnehotCrossEntropyLoss(reduction=self.reduction)
        elif self.criterion_type == 'label_smoothing':
            self.criterion=LabelSmoothingLoss(num_classes=self.num_classes,smoothing=smoothing,reduction=self.reduction)

    def forward(self, prediction, target, target_b=None, lam_a=1.0, lam_b=1.0, configs=None):
        if self.criterion_type == 'crossentropy':
            if target_b is None:
                loss = self.criterion(prediction, target)
            else:
                loss_a = self.criterion(prediction, target)
                loss_b = self.criterion(prediction, target_b)
                loss = lam_a * loss_a + lam_b * loss_b
        elif self.criterion_type == 'bce':
            if target_b is None:
                loss = self.criterion(prediction, target)
                return loss
            else:
                prob=torch.softmax(prediction,dim=1)
                if not type(lam_a,float):
                    if len(lam_a.shape)==1:
                        lam_a=lam_a.unsqueeze(1)
                        lam_b=lam_b.unsqueeze(1)
                if target.shape!=prediction.shape:
                    onehot_target=F.one_hot(target,self.num_classes).float()
                    onehot_target_b=F.one_hot(target_b,self.num_classes).float()
                else:
                    onehot_target=target
                    onehot_target_b=target_b
                mixed_target=lam_a*onehot_target+lam_b*onehot_target_b
                loss = self.criterion(prob, mixed_target)
        elif self.criterion_type == 'onehot_ce':
            if target_b is None:
                onehot_target=F.one_hot(target,self.num_classes).float()
                loss = self.criterion(prediction, onehot_target)
            else:
                if not type(lam_a,float):
                    if len(lam_a.shape)==1:
                        lam_a=lam_a.unsqueeze(1)
                        lam_b=lam_b.unsqueeze(1)
                if target.shape!=prediction.shape:
                    onehot_target=F.one_hot(target,self.num_classes).float()
                    onehot_target_b=F.one_hot(target_b,self.num_classes).float()
                else:
                    onehot_target=target
                    onehot_target_b=target_b
                mixed_target=lam_a*onehot_target+lam_b*onehot_target_b
                loss = self.criterion(prediction, mixed_target)
        elif self.criterion_type == 'label_smoothing':
            if target_b is None:
                loss = self.criterion(prediction, target)
            else:
                loss_a = self.criterion(prediction, target)
                loss_b = self.criterion(prediction, target_b)
                loss = lam_a * loss_a + lam_b * loss_b
        else:
            raise ValueError('criterion type {} is not supported'.format(self.criterion_type))
        
        if self.criterion_type=='bce' and self.reduction == 'mean':
            loss = loss.mean(dim=1)
        elif self.reduction == 'none':
            loss = loss.mean()
        return loss
            