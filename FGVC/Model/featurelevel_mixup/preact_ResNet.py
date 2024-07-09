from Mixup.featurelevel_mixup import mixup_process
from Model.featurelevel_mixup.layer_mix import determine_layer_mix
from Model.preact_ResNet import PreActResNet
import random, copy, torch
import torch.nn as nn


class PreActResNet_Feature(PreActResNet):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs=configs
            

    def forward(self, x, y=None, mixup=False, sc=None, mixup_indices=None, get_mixed_feature=False, configs=None,extract_feature=False):
        if mixup==True:
            return_verboses={}
            verboses={}
            layer_mix= determine_layer_mix(train_mode=self.configs['train_mode'],layer_mix=self.configs['layer_mix'], configs=configs)

            out = x
            y_a=y

            out = self.conv1(out)
            if self.residual_len == 4:
                if self.maxpool:
                    out = self.maxpool(out)
                if layer_mix == 1:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                    
                out = self.layer1(out)
                if layer_mix == 2:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                    
                out = self.layer2(out)
                if layer_mix == 3:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                    
                out = self.layer3(out)
                if layer_mix == 4:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                    
                out = self.layer4(out)
                # out = self.bn(out)
                # out = self.relu(out)
                if layer_mix == 5:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                    

                if layer_mix > 5:
                    raise ValueError('layer_mix should be less than 6')

            elif self.residual_len == 3:
                if layer_mix == 1:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                out = self.layer1(out)
                if layer_mix == 2:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                out = self.layer2(out)
                if layer_mix == 3:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                out = self.layer3(out)
                # out = self.bn(out)
                # out = self.relu(out)
                if layer_mix == 4:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
                if layer_mix > 4:
                    raise ValueError('layer_mix should be less than 5')
                out = self.avgpool(out)

            out = self.avgpool(out)
            out = out.view(out.size(0), -1)

            if layer_mix == -1:
                out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix, sc=sc,mixup_indices=mixup_indices, configs=configs,c_param=self.c_layer_param,s_param=self.s_layer_param,verboses=verboses)
            out = self.linear(out)
            if verboses:
                return_verboses.update(verboses)
            return out, y_a, y_b, lam_a, lam_b, return_verboses
        else: # non mixup (in test time)
            return super().forward(x,extract_feature=extract_feature)
