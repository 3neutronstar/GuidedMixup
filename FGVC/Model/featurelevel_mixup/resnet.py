from Mixup.featurelevel_mixup import mixup_process
from Model.featurelevel_mixup.layer_mix import determine_layer_mix
from Model.resnet import ResNet
import numpy as np
import random
import torch,copy
import torch.nn as nn
import torch.nn.functional as F

class ResNet_Feature(ResNet):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs=configs

        # block
        if self.residual_len==4:
            self.layer_blocks=[self.layer1, self.layer2, self.layer3, self.layer4]
        elif self.residual_len==3:
            self.layer_blocks=[self.layer1, self.layer2, self.layer3]
        else: raise NotImplementedError

    def forward(self, x, y=None, mixup=False, sc=None, mixup_indices=None, get_mixed_feature=False, configs=None, extract_feature=False, retain_grad=False):
        if '-ap' in self.configs['train_mode']:
            retain_grad=True
        return_verboses={}
        verboses={}
        
        if mixup==True:
            layer_mix = determine_layer_mix(train_mode=self.configs['train_mode'],layer_mix=self.configs['layer_mix'], configs=configs)
            out = x
            y_a=y

            out = self.relu(self.bn1(self.conv1(out)))
            if self.residual_len == 4:
                if self.maxpool:
                    out = self.maxpool(out)
            layer_idx=0
            for layer in self.layer_blocks:
                if layer_mix == layer_idx+1:
                    # mixup process
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                out = layer(out)
                layer_idx+=1
            if layer_mix == len(self.layer_blocks)+1:
                out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
            # layer mix
            if layer_mix > len(self.layer_blocks)+1:
                raise ValueError('layer_mix should be less than {}'.format(len(self.layer_blocks)+1))

            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            if layer_mix == -1:
                out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                
            out = self.fc(out)
            return_verboses['layer_mix']=layer_mix
            if verboses is not None:
                return_verboses.update(verboses)
            # if configs['train_mode']=='channel_drop':
            #     self.train()
            return out, y_a, y_b, lam_a, lam_b, return_verboses
        
        elif mixup==False and configs is not None: # training without augmentation (for analysis)
            features=[]
            out = x
            out = self.relu(self.bn1(self.conv1(out)))
            features.append(out.detach())
            if self.residual_len == 4 and self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            features.append(out.detach())
            out = self.layer2(out)
            features.append(out.detach())
            out = self.layer3(out)
            features.append(out.detach())
            if self.residual_len == 4:
                out = self.layer4(out)
                features.append(out.detach())
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out,features
        else: # non mixup (in test time)
            return super().forward(x, extract_feature=extract_feature,retain_grad=retain_grad)