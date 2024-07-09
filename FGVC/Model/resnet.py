import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from Model.layer.batchnorm import CustomBatchNorm2d,  CustomBatchNorm2d_depreciated



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(inplanes: int, out_planes: int, stride: int=1, groups: int = 1, dilation: int = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,groups=groups,dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.identity=nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += self.identity(identity)
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample= None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.identity=nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += self.identity(identity)
        out = self.relu(out)

        return out

class ResNet(nn.Module): # Follow CutMix
    model_dict = {'resnet18': (BasicBlock,[2, 2, 2, 2]),
                    'resnet34': (BasicBlock,[3, 4, 6, 3]),
                    'resnet50': (Bottleneck,[3, 4, 6, 3]),
                    "resnext50_32x4d":(Bottleneck,[3, 4, 6, 3],32,4),
                    "resnext101_32x8d":(Bottleneck,[3, 4, 23, 3],32,8),
                    "resnext101_64x4d":(Bottleneck,[3, 8, 36, 3],64,4),
                    'resnet101': (Bottleneck,[3, 4, 23, 3]),
                    'resnet152': (Bottleneck,[3, 8, 36, 3]),
                    'resnet20': (BasicBlock,[3, 3, 3]),
                    'resnet32': (BasicBlock,[5, 5, 5]),
                    'resnet44': (BasicBlock,[7, 7, 7]),
                    'resnet56': (BasicBlock,[9, 9, 9]),
                    'resnet110': (BasicBlock,[18, 18, 18]),
                    'resnet1202': (BasicBlock,[200, 200, 200])
                    }
    def __init__(self, configs):
        super(ResNet, self).__init__()
        num_classes = configs['num_classes']
        block, num_blocks = self.model_dict[configs['model']][0], self.model_dict[configs['model']][1]
        self.residual_len=len(num_blocks)
        self.dilation = 1

        norm_layer=nn.BatchNorm2d
        if configs['mode'] =='train' and configs['train_mode']=='channel_drop':
            norm_layer=CustomBatchNorm2d
            # norm_layer= CustomBatchNorm2d_depreciated


        if 'resnext' in configs['model']:
            self.base_width = self.model_dict[configs['model']][2] #width per group
            self.groups = self.model_dict[configs['model']][3]
        else:
            self.base_width = 64
            self.groups=1
        
        if self.residual_len == 4:
            replace_stride_with_dilation = [False, False, False]
            self.inplanes=64
            if configs['dataset'] in ['cifar10','cifar100']:
                self.maxpool=None
                stride=1
                self.avgpool = nn.AvgPool2d(4)
                self.conv1 = nn.Conv2d(3,
                                    self.inplanes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
            elif configs['dataset'] in ['tiny-imagenet'] and not configs['eval_loc']:
                self.maxpool=None
                stride=2
                self.avgpool = nn.AvgPool2d(4)
                self.conv1 = nn.Conv2d(3,
                                    self.inplanes,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
            else: # size 224
                stride=2
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = nn.AdaptiveAvgPool2d((1,1))#nn.AvgPool2d(7)
                self.conv1 = nn.Conv2d(
                    3, self.inplanes, kernel_size=7, stride=stride, padding=3, bias=False)
                
            self.bn1 = norm_layer(self.inplanes)
            stride_list = [1, 2, 2, 2]

            self.layer1 = self._make_layer(
                block, 64, num_blocks[0], stride=stride_list[0],norm_layer=norm_layer)
            self.layer2 = self._make_layer(
                block, 64*2, num_blocks[1], stride=stride_list[1],dilate=replace_stride_with_dilation[0],norm_layer=norm_layer)
            self.layer3 = self._make_layer(
                block, 64*4, num_blocks[2], stride=stride_list[2],dilate=replace_stride_with_dilation[1],norm_layer=norm_layer)
            self.layer4 = self._make_layer(
                block, 64*8, num_blocks[3], stride=stride_list[3],dilate=replace_stride_with_dilation[2],norm_layer=norm_layer)
            self.fc = nn.Linear(
                64*8*block.expansion, num_classes)
            self.last_feature_dim=self.inplanes*8*block.expansion
        elif self.residual_len == 3: # for cifar dataset
            self.inplanes = 16

            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.avgpool = nn.AvgPool2d(8)
            stride_list = [1, 2, 2]
            self.layer1 = self._make_layer(
                block, 16, num_blocks[0], stride=stride_list[0],norm_layer=norm_layer)
            self.layer2 = self._make_layer(
                block, 16*2, num_blocks[1], stride=stride_list[1],norm_layer=norm_layer)
            self.layer3 = self._make_layer(
                block, 16*4, num_blocks[2], stride=stride_list[2],norm_layer=norm_layer)
            self.fc = nn.Linear(
                16*4*block.expansion, num_classes)
            self.last_feature_dim=self.inplanes*4*block.expansion
            
        self.relu=nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilate=False,norm_layer=nn.BatchNorm2d):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.groups,self.base_width,previous_dilation,norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, extract_feature=False, retain_grad=False):
        if extract_feature:
            return self.extract_feature(x,retain_grad=retain_grad)
        if self.residual_len == 4:
            out = self.relu(self.bn1(self.conv1(x)))
            if self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
        elif self.residual_len == 3:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def extract_feature(self, x, retain_grad=False):
        feature = []
        out = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool:
            out=self.maxpool(out)
            if retain_grad:
                out.retain_grad()
        feature.append(out)
        out = self.layer1(out)
        if retain_grad:
            out.retain_grad()
        feature.append(out)
        out = self.layer2(out)
        if retain_grad:
            out.retain_grad()
        feature.append(out)
        out = self.layer3(out)
        if retain_grad:
            out.retain_grad()
        feature.append(out)
        if self.residual_len==4:
            out = self.layer4(out)
            if retain_grad:
                out.retain_grad()
            feature.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature

class ResNet_ft(ResNet):
    def __init__(self, configs):
        super().__init__(configs)
    
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        feature = []
        if self.residual_len==4:
            if self.maxpool:
                out=self.maxpool(out)
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
            out = self.layer4(out)
            feature.append(out)

        elif self.residual_len==3:
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
        out = self.avgpool(out)
        feature.append(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, feature
