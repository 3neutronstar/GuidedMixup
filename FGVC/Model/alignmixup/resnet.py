'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mixup.featurelevel_mixup import mixup_process
from Model.featurelevel_mixup.layer_mix import determine_layer_mix
# from Model.preact_ResNet import PreActBlock, PreActBottleneck # its own preact resnet

from Model.resnet import BasicBlock, Bottleneck

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dataset=None):
        super(ResNet, self).__init__()
        self.maxpool = None
        self.in_planes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.relu=nn.ReLU(inplace=True)
        if dataset.startswith('cifar'):
            self.avgpool = nn.AvgPool2d(4)
        else:
            self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class ResNetClassifier(ResNet):
    def __init__(self, block, num_blocks, num_classes=10,dataset=None):
        super(ResNetClassifier, self).__init__(block, num_blocks, dataset=dataset)
        self.fc = nn.Linear(512*block.expansion, num_classes)
    def forward(self, x, y=None, mixup=False, sc=None, mixup_indices=None,configs=None,extract_feature=False):
        if configs is not None:
            if mixup==True:
                layer_mix=determine_layer_mix(train_mode=configs['train_mode'],layer_mix=configs['layer_mix'], configs=configs)

                out = x
                y_a=y

                if layer_mix == 0:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)

                out = self.relu(self.bn1(self.conv1(out)))
                if self.maxpool:
                    out = self.maxpool(out)
                if layer_mix == 1:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)
                out = self.layer1(out)
                if layer_mix == 2:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)
                out = self.layer2(out)
                if layer_mix == 3:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)
                out = self.layer3(out)
                if layer_mix == 4:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)
                out = self.layer4(out)
                if layer_mix == 5:
                    out, y_a, y_b, lam_a, lam_b, verboses = mixup_process(x,out, y, layer_mix,sc=sc,indices=mixup_indices,configs= configs)
                elif layer_mix >=6:
                    raise ValueError('layer_mix should be 0~5')
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)

                return out, y_a, y_b, lam_a, lam_b, verboses
            if configs['train_mode'] in ['train_verbose','ci_sim']:
                out = x
                features=[]
                out = self.relu(self.bn1(self.conv1(out)))
                features.append(out.detach())
                if self.maxpool:
                    out = self.maxpool(out)
                out = self.layer1(out)
                features.append(out.detach())
                out = self.layer2(out)
                features.append(out.detach())
                out = self.layer3(out)
                features.append(out.detach())
                out = self.layer4(out)
                features.append(out.detach())
                out = self.avgpool(out)
                out = out.view(out.size(0), -1)
                out = self.fc(out)
                return out, features
        else: # non mixup (in test time)
            out = super(ResNetClassifier, self).forward(x,extract_feature=extract_feature)
            # out = self.avgpool(out)
            # out = out.view(out.size(0), -1)
            # out = self.fc(out)
            return out
        

# def ResNet18(num_classes):
#     return ResNet(BasicBlock, [2,2,2,2])

# def ResNet34(num_classes):
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50(num_classes):
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet101(num_classes):
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet152(num_classes):
#     return ResNet(Bottleneck, [3,8,36,3])

def PreActResNet18(num_classes, dataset=None):
    return ResNetClassifier(PreActBlock, [2,2,2,2], num_classes, dataset=dataset)

def PreActResNet34(num_classes, dataset=None):
    return ResNetClassifier(PreActBlock, [3,4,6,3], num_classes, dataset=dataset)

def PreActResNet50(num_classes, dataset=None):
    return ResNetClassifier(PreActBottleneck, [3,4,6,3], num_classes, dataset=dataset)

def PreActResNet101(num_classes, dataset=None):
    return ResNetClassifier(PreActBottleneck, [3,4,23,3], num_classes, dataset=dataset)

def PreActResNet152(num_classes, dataset=None):
    return ResNetClassifier(PreActBottleneck, [3,8,36,3], num_classes, dataset=dataset)

def get_alignmixup_preact_resnet(configs):
    alignmixup_model='_'.join(configs['model'].split('_')[1:])
    assert configs['model'].split('_')[0]=='alignmixup'
    assert configs['dataset'] in ['cifar10' ,'cifar100','tiny-imagenet']
    resnet_dict={
        'preact_resnet18': PreActResNet18,
        'preact_resnet34': PreActResNet34,
        'preact_resnet50': PreActResNet50,
        'preact_resnet101': PreActResNet101,
        'preact_resnet152': PreActResNet152,
    }
    try:
        return resnet_dict[alignmixup_model](num_classes=configs['num_classes'], dataset=configs['dataset'])
    except:
        raise ValueError('In AlignMixup model name should be one of {}, Yours: {}'.format(resnet_dict.keys(),alignmixup_model))