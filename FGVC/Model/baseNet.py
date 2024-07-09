from Model.preact_ResNet import Preact_ResNet_ft
from Model.resnet import ResNet_ft
from Model.preact_ResNet import PreActResNet
from Model.resnet import ResNet

def get_model(configs):
    if configs['mixup_type'] in ['hidden']:
        if 'alignmixup' in configs['model']: # alignmixup preactresnet
            from Model.alignmixup.resnet import get_alignmixup_preact_resnet
            model=get_alignmixup_preact_resnet(configs)
        elif 'preact_resnet' in configs['model']:
            from Model.featurelevel_mixup.preact_ResNet import PreActResNet_Feature
            model=PreActResNet_Feature(configs)
        elif 'resnet' in configs['model'] or 'resnext' in configs['model']:
            from Model.featurelevel_mixup.resnet import ResNet_Feature
            pretrained=False
            if 'pretrained' in configs['model']:
                configs['model']=configs['model'][11:]
                pretrained=True
                num_classes=configs['num_classes']
                configs['num_classes']=1000
            model = ResNet_Feature(configs)
        elif 'densenet' in configs['model']:
            from Model.featurelevel_mixup.densenet import densenet_feature
            model = densenet_feature(configs)
        else:
            raise NotImplementedError('model not implemented in hidden')
    else:
        if 'alignmixup' in configs['model']: # alignmixup preactresnet
            from Model.alignmixup.resnet import get_alignmixup_preact_resnet
            model=get_alignmixup_preact_resnet(configs)
        elif "preact_resnet" in configs["model"]:
            model = PreActResNet(configs)

        elif "resnet" in configs["model"]:
            pretrained=False
            if 'pretrained' in configs['model']:
                configs['model']=configs['model'][11:]
                pretrained=True
                num_classes=configs['num_classes']
                configs['num_classes']=1000
            model=ResNet(configs)

        elif "densenet" in configs["model"]:
            from Model.densenet import densenet
            model=densenet(configs)

        else:
            print("No Model")
            raise NotImplementedError

        if configs['mode']=='train' and configs['train_mode']in ['train_verbose'] and 'resnet' in configs['model']:
            if isinstance(model, PreActResNet):
                model=Preact_ResNet_ft(configs)
            elif isinstance(model, ResNet):
                model=ResNet_ft(configs)


    if 'pre' not in configs['model'] and 'resnet' in configs['model'] and pretrained:
        configs['num_classes']=num_classes
        from torch.hub import load_state_dict_from_url
        model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
            }
        state_dict = load_state_dict_from_url(model_urls[configs['model']])
        if configs['mode']=='eval':
            try:
                model.load_state_dict(state_dict)
            except:
                pass
        else:
            model.load_state_dict(state_dict)
        if configs['mode']!='eval':
            model.fc = model.fc.__class__(
                model.fc.weight.size(1), configs["num_classes"]
                )
        elif configs['mode']=='eval' and configs['evaluator']=='augmentation' and configs['dataset']!='imagenet':
            model.fc = model.fc.__class__(
                model.fc.weight.size(1), configs["num_classes"]
                )
        else:
            pass
        configs['model']='pretrained_'+configs['model']
        print(model.fc.weight.shape)
    return model
