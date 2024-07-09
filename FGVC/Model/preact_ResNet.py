'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from Model.layer.batchnorm import CustomBatchNorm2d


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,norm_layer=None):
        super(PreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        self.bn1 = norm_layer(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,norm_layer=None):
        super(PreActBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        self.bn1 = norm_layer(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module): # follow PuzzleMix
    model_dict = {
        'preact_resnet18': (PreActBlock, [2, 2, 2, 2]),
        'preact_resnet34': (PreActBlock, [3, 4, 6, 3]),
        'preact_resnet50': (PreActBottleneck, [3, 4, 6, 3]),
        'preact_resnet101': (PreActBottleneck, [3, 4, 23, 3]),
        'preact_resnet152': (PreActBottleneck, [3, 4, 36, 3]),
        'preact_resnet200': (PreActBottleneck, [3, 8, 36, 3]),
        'preact_resnet20': (PreActBlock, [3, 3, 3]),
        'preact_resnet32': (PreActBlock, [5, 5, 5]),
        'preact_resnet44': (PreActBlock, [7, 7, 7]),
        'preact_resnet56': (PreActBlock, [9, 9, 9]),
        'preact_resnet110': (PreActBlock, [18, 18, 18]),
        'preact_resnet1202': (PreActBlock, [200, 200, 200])
    }

    def __init__(self, configs):
        super(PreActResNet, self).__init__()
        num_classes = configs['num_classes']
        block, num_blocks = self.model_dict[configs['model']]
        self.residual_len=len(num_blocks)
        norm_layer=nn.BatchNorm2d
        if configs['train_mode']=='channel_drop':
            norm_layer=CustomBatchNorm2d

        if self.residual_len == 4:
            initial_channels = 64
            self.in_planes=initial_channels
            if configs['dataset'] in ['cifar10','cifar100']:
                self.maxpool=None
                stride=1
                self.conv1 = nn.Conv2d(3,
                                    initial_channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
                self.avgpool = nn.AvgPool2d(4)
            elif configs['dataset'] in ['tiny-imagenet']:
                self.maxpool=None
                stride=2
                self.avgpool = nn.AvgPool2d(4)
                self.conv1 = nn.Conv2d(3,
                                    initial_channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
            else: # size 224
                stride=2
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = nn.AvgPool2d(7)
                self.conv1 = nn.Conv2d(
                    3, initial_channels, kernel_size=7, stride=stride, padding=3, bias=False)
                
            stride_list = [1, 2, 2, 2]

            self.layer1 = self._make_layer(
                block, initial_channels, num_blocks[0], stride=stride_list[0],norm_layer=norm_layer)
            self.layer2 = self._make_layer(
                block, initial_channels*2, num_blocks[1], stride=stride_list[1],norm_layer=norm_layer)
            self.layer3 = self._make_layer(
                block, initial_channels*4, num_blocks[2], stride=stride_list[2],norm_layer=norm_layer)
            self.layer4 = self._make_layer(
                block, initial_channels*8, num_blocks[3], stride=stride_list[3],norm_layer=norm_layer)
            self.linear = nn.Linear(
                initial_channels*8*block.expansion, num_classes)
            # TODO TEMP
            self.relu = nn.ReLU(inplace=True)
            # self.bn = nn.BatchNorm2d(initial_channels*8*block.expansion)
            # self.last_feature_dim=initial_channels*8*block.expansion

        elif self.residual_len == 3:
            initial_channels = 16
            self.in_planes=initial_channels

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.avgpool = nn.AvgPool2d(8)
            stride_list = [1, 2, 2]
            self.layer1 = self._make_layer(
                block, initial_channels, num_blocks[0], stride=stride_list[0],norm_layer=norm_layer)
            self.layer2 = self._make_layer(
                block, initial_channels*2, num_blocks[1], stride=stride_list[1],norm_layer=norm_layer)
            self.layer3 = self._make_layer(
                block, initial_channels*4, num_blocks[2], stride=stride_list[2],norm_layer=norm_layer)
            self.linear = nn.Linear(
                initial_channels*4*block.expansion, num_classes)
            # TODO TEMP
            # self.bn = nn.BatchNorm2d(initial_channels*4*block.expansion)
            # self.relu = nn.ReLU(inplace=True)
            self.last_feature_dim=initial_channels*4*block.expansion

        self.relu = nn.ReLU(inplace=True)
        if self.residual_len ==3:
            for m in self.modules(): # as shown in comixup / PuzzleMix
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride,norm_layer=nn.BatchNorm2d):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,norm_layer=norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,extract_feature=False):
        if extract_feature:
            return self.extract_feature(x)
        if self.residual_len == 4:
            # out = self.conv1(x)
            out = self.conv1(x)
            if self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            # out = self.bn(out)
            # out = self.relu(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.residual_len == 3:
            # out = self.conv1(x)
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            # out = self.bn(out)
            # out = self.relu(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        return out

    def extract_feature(self,x):
        out = self.conv1(x)
        feature = []
        if self.residual_len==4:
            # out=self.maxpool(out)
            if self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
            out = self.layer4(out)
            # out = self.bn(out)
            # out = self.relu(out)
            feature.append(out)
            out = F.avg_pool2d(out,4)
        elif self.residual_len==3:
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            # out = self.bn(out)
            # out = self.relu(out)
            feature.append(out)
            out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, feature

class Preact_ResNet_ft(PreActResNet):
    def __init__(self, configs):
        super().__init__(configs)

    def forward(self,x):
        out = self.conv1(x)
        feature = []
        if self.residual_len==4:
            # out=self.maxpool(out)
            if self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
            out = self.layer4(out)
            # out = self.bn(out)
            # out = self.relu(out)
            feature.append(out)
            out = F.avg_pool2d(out,4)
        elif self.residual_len==3:
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            # out = self.bn(out)
            # out = self.relu(out)
            feature.append(out)
            out = self.avgpool(out)
        feature.append(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, feature
