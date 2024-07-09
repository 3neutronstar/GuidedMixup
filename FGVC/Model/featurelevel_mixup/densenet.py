import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Mixup.featurelevel_mixup import mixup_process
from Model.densenet import _DenseBlock, model_urls, _DenseBlock, _Transition,_load_state_dict
import random

from Model.featurelevel_mixup.layer_mix import determine_layer_mix

class DenseNet_Feature(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False,configs=None):
        super(DenseNet_Feature, self).__init__()
        if configs:
            self.configs=configs
            
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.layers=[]
        for i, num_layers in enumerate(block_config):
            seq=nn.Sequential()
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            # self.features.add_module('denseblock%d' % (i + 1), block)
            seq.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                seq.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            self.layers.append(seq)
            self.add_module('denseblock%d' % (i + 1), seq)

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.norm5=nn.BatchNorm2d(num_features)

        # Linear layer
        self.last_feature_dim=num_features
        self.linear = nn.Linear(num_features, num_classes) # original: classifier / linear -> change for the SnapMix

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
        self.avgpool=nn.AvgPool2d(7)

    def forward(self, x, y=None, mixup=False, sc=None, mixup_indices=None, get_mixed_feature=False, configs=None,extract_feature=False):
        if extract_feature:
            raise NotImplementedError("Not implemented yet using DenseNet")
        if mixup==True:
            return_verboses={}
            verboses={}
            layer_mix = determine_layer_mix(train_mode=self.configs['train_mode'],layer_mix=self.configs['layer_mix'], configs=configs)

            out = x
            y_a=y
            if layer_mix == 0 or configs['train_mode'] in ['reinforce-sr','reinforce-ap']:
                out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc, mixup_indices=mixup_indices, configs= configs,verboses=verboses)
                if configs['train_mode'] in ['reinforce-sr','reinforce-ap']:
                    temp_y_b,temp_lam_a,temp_lam_b=y_b,lam_a,lam_b

            out=self.features(out)

            l=1 # layer index
            for layer in self.layers:
                if l==layer_mix:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc, mixup_indices=mixup_indices, configs= configs, verboses=verboses)
                out = layer(out)
                l+=1
            
            if l==layer_mix:
                out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc, mixup_indices = mixup_indices, configs = configs, verboses=verboses)
            elif layer_mix>l :
                raise ValueError('layer_mix is too large')
            
            out=self.norm5(out)
            out = F.relu(out, inplace=True)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.linear(out) # classifier

            if verboses:
                return_verboses.update(verboses)
            if configs['train_mode'] in ['reinforce-sr','reinforce-ap']:
                y_b,lam_a,lam_b=temp_y_b,temp_lam_a,temp_lam_b
            
            return out, y_a, y_b, lam_a, lam_b, return_verboses

        else:
            features=self.features(x)
            for layer in self.layers:
                features = layer(features)
            features=self.norm5(features)
            out = F.relu(features, inplace=True)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.linear(out) # classifier
            return out


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,train_mode='baseline',
              **kwargs):
    model = DenseNet_Feature(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)


def densenet265(pretrained=False, progress=True, **kwargs):
    r"""Densenet-265 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet265', 32, (6, 12, 64, 48), 64, pretrained, progress,
                     **kwargs)


def densenet_feature(configs):
    dense_dict = {
        'densenet121': densenet121,
        'pretrained_densenet121': densenet121,
        'densenet161': densenet161,
        'pretrained_densenet161': densenet161,
        'densenet169': densenet169,
        'pretrained_densenet169': densenet169,
        'densenet201': densenet201,
        'pretrained_densenet201': densenet201,
        'densenet265': densenet265,
         }
    if 'pretrained' in configs['model']:
        pretrained=True
        model=dense_dict[configs['model']](pretrained)        
        model.classifier = model.classifier.__class__(
            model.classifier.weight.size(1), configs["num_classes"]
        )
    else:
        pretrained=False
        model=dense_dict[configs['model']](pretrained,num_classes=configs['num_classes'],train_mode=configs['train_mode'],configs=configs)
    return model
