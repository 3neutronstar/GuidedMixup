import os
import copy
import numpy as np
import sys
from functools import reduce
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from operator import __or__

sys.path.append('.')
from DataLoad.aircraft import Aircraft
from DataLoad.cars196 import Cars196Dataset
from DataLoad.cub200 import Cub2011
import torch
from torch.utils.data.dataset import Subset
from DataLoad.customdataset import IndexDataset, TransformTensorDataset
import torchvision
from DataLoad.cutout import Cutout
from torchvision import datasets
import torchvision.transforms as transforms
from DataLoad.flowers102 import Flower102
from DataLoad.inat2017 import INat2017
from Utils.augmentation import *
import os
import copy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class ChangeAugmentation:
    def __init__(self, loader):
        self.original_transform = copy.deepcopy(loader.dataset.transform)

    def __call__(self, loader, transform_list: list = None):
        if transform_list is None:
            return loader
        else:
            return self.run(loader, transform_list)

    def run(self, loader, transform_list: list = None) -> DataLoader:
        loader.dataset.transform = copy.deepcopy(self.original_transform)
        for transform in reversed(transform_list):
            loader.dataset.transform.transforms.insert(0, transform)
        return loader


class DatasetLoader:
    def __init__(self, dataset_path=None, configs=None) -> None:
        self.configs = configs
        self.dataset_path = dataset_path

        ## Normalize Mean & Std ##
        if configs['dataset'] == 'cifar100':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif configs['dataset'] == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif configs['dataset'] == 'tiny-imagenet':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]
        elif configs['dataset'] in ['dogs120','cub200','cars196','caltech101','caltech256','flowers102','aircraft100','food101','caltech101','caltech256','inat','celeba','waterbirds']:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if configs['fgvc_resize']:
                resize_shape=512 # 한개로 해야 ratio로 자연스럽게 줄어들음
                crop_shape=448
            else:
                resize_shape=256 # 한개로 해야 ratio로 자연스럽게 줄어들음
                crop_shape=224
            print("Resize shape: {} crop shape: {}".format(resize_shape, crop_shape))
        else:
            raise ValueError('Dataset {} not supported'.format(configs['dataset']))
        configs['mean'] = torch.tensor(
            mean, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        configs['std'] = torch.tensor(
            std, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        ##########################


        if configs['dataset'] == 'cifar100':
            normalize = transforms.Normalize(mean=mean, std=std)
            self.train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        elif configs['dataset'] == 'cifar10':
            normalize = transforms.Normalize(
                mean=mean, std=std)
            self.train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        elif configs['dataset'] == 'celeba':
            '''
            https://github.com/alinlab/lookahead_pruning/blob/master/dataset.py
            '''
            # normalize = transforms.Normalize(mean=mean,
            #                                  std=std)
            # self.train_transform = transforms.Compose([
            #     transforms.RandomCrop(64,4),
            #     transforms.RandomHorizontalFlip(0.5),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            # self.test_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            self.train_transforms = transforms.Compose([ #https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/celeba/train.py
                transforms.Resize(size=resize_shape),
                transforms.RandomCrop(size=crop_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
            self.test_transforms = transforms.Compose([
                transforms.Resize(size=resize_shape),
                transforms.CenterCrop(size=crop_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
        elif configs['dataset'] == 'tiny-imagenet':
            '''
            https://github.com/alinlab/lookahead_pruning/blob/master/dataset.py
            '''
            normalize = transforms.Normalize(mean=mean,
                                             std=std)
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64,4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize(256),transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            if 'tiny_resize' in self.configs.keys():
                if self.configs['tiny_resize']:
                    '''
                    https://github.com/pytorch/examples/blob/master/imagenet/main.py
                    '''
                    normalize = transforms.Normalize(mean=mean,
                                                    std=std)

                    self.train_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
                    self.test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ])
        elif configs['dataset'] == 'stl10':
            self.train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(96),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif configs['dataset'] in ['caltech256', 'caltech101']:
            normalize = transforms.Normalize(
                mean, std)
            def tmp_func(x):
                return x.convert('RGB')
            self.train_transform = transforms.Compose([
                transforms.Lambda(tmp_func),
                transforms.Resize(resize_shape),
                transforms.RandomCrop(crop_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                normalize])
            self.test_transform = transforms.Compose([
                transforms.Lambda(tmp_func),
                transforms.Resize(resize_shape),
                transforms.CenterCrop(crop_shape),
                transforms.ToTensor(),
                normalize
            ])
            configs['num_workers'] = 0  # error fix
        elif configs['dataset'] in ['cars196','dogs120','aircraft100','flowers102','cub200','food101','inat']:
            normalize = transforms.Normalize(
                mean=mean, std=std)
            self.train_transform = \
                transforms.Compose([
                    transforms.Resize(resize_shape),
                    transforms.RandomCrop(crop_shape),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            self.test_transform = transforms.Compose([
                transforms.Resize(resize_shape),
                transforms.CenterCrop(crop_shape),
                transforms.ToTensor(),
                normalize,
            ])
        if configs['augmix'] == True:
            try:
                self.train_transform.transforms.append(transforms.AugMix())
            except:
                print('[WARNING] AugMix can not be used in PyTorch version {}. Please upgrade torchvision and pytorch.'.format(torch.__version__))
        if configs['randaug'] == True:
            try:
                self.train_transform.transforms.append(transforms.RandAugment())
            except:
                print('[WARNING] RandAug can not be used in PyTorch version {}. Please upgrade torchvision and pytorch.'.format(torch.__version__))
        if configs['cutout'] == True:
            self.train_transform.transforms.append(
                Cutout(n_holes=configs['cutout_holes'], length=configs['cutout_length']))
            # self.test_transform.transforms.append(Cutout(n_holes=configs['cutout_holes'], length=configs['cutout_length']))

    def get_dataset(self):
        if self.dataset_path is None:
            if sys.platform == 'linux':
                dataset_path = '/data'
            elif sys.platform == 'win32':
                dataset_path = '..\data\dataset'
            else:
                dataset_path = '~/dataset'
        else:
            dataset_path = self.dataset_path # parent directory
                
        dataset_path = os.path.join(dataset_path, self.configs['dataset'])

        if self.configs['dataset'] == 'cifar100':
            train_data = datasets.CIFAR100(root=dataset_path, train=True,
                                           download=True, transform=self.train_transform)
            test_data = datasets.CIFAR100(root=dataset_path, train=False,
                                          download=False, transform=self.test_transform)

        elif self.configs['dataset'] == 'cifar10':
            train_data = datasets.CIFAR10(root=dataset_path, train=True,
                                          download=True, transform=self.train_transform)
            test_data = datasets.CIFAR10(root=dataset_path, train=False,
                                         download=False, transform=self.test_transform)

        elif self.configs['dataset'] == 'tiny-imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val')
            train_data = torchvision.datasets.ImageFolder(
                root=traindata_save_path, transform=self.train_transform)
            test_data = torchvision.datasets.ImageFolder(
                root=testdata_save_path, transform=self.test_transform)

        elif self.configs['dataset'] == 'stl10':
            train_data = torchvision.datasets.STL10(
                root=dataset_path, split="train", transform=self.train_transform, download=True)
            test_data = torchvision.datasets.STL10(
                root=dataset_path, split="test", transform=self.train_transform, download=False)

        elif self.configs['dataset'] == 'caltech101':
            data = torchvision.datasets.Caltech101(
                root=dataset_path, download=True)
            train_indices, test_indices = train_test_split(torch.arange(len(
                data.y)), stratify=data.y, test_size=0.2, train_size=0.8, random_state=self.configs['seed'])
            train_data = Subset(data, train_indices)
            test_data = Subset(data, test_indices)
            train_data = TransformTensorDataset(
                train_data, self.train_transform)
            test_data = TransformTensorDataset(test_data, self.test_transform)

        elif self.configs['dataset'] == 'caltech256':
            dataset_path = os.path.join(
                dataset_path, '256_ObjectCategories')
            # data=torchvision.datasets.Caltech256(root=dataset_path,download=True)
            data = torchvision.datasets.ImageFolder(root=dataset_path)
            train_indices, test_indices = train_test_split(torch.arange(len(
                data.targets)), stratify=data.targets, test_size=0.2, train_size=0.8, random_state=self.configs['seed'])
            train_data = Subset(data, train_indices)
            test_data = Subset(data, test_indices)
            train_data = TransformTensorDataset(
                train_data, self.train_transform)
            test_data = TransformTensorDataset(test_data, self.test_transform)
        elif self.configs['dataset'] == 'cub200':
            train_data = Cub2011(dataset_path, train=True,
                                 transform=self.train_transform, download=True)
            test_data = Cub2011(dataset_path, train=False,
                                transform=self.test_transform, download=False)
        elif self.configs['dataset'] == 'cars196':
            train_data = Cars196Dataset(
                root=dataset_path, train=True, transform=self.train_transform)
            test_data = Cars196Dataset(
                root=dataset_path, train=False, transform=self.test_transform)
        elif self.configs['dataset'] == 'aircraft100':
            train_data = Aircraft(dataset_path, train=True,
                                  download=False, transform=self.train_transform)
            test_data = Aircraft(dataset_path, train=False,
                                 download=False, transform=self.test_transform)
        elif self.configs['dataset']=='flowers102':
            train_data=Flower102(dataset_path,mode='train',transform=self.train_transform)
            test_data=Flower102(dataset_path,mode='val',transform=self.test_transform)
        elif self.configs['dataset']=='inat':
            train_data=INat2017(dataset_path,'train',transform=self.train_transform,download=True)
            test_data=INat2017(dataset_path,'val',transform=self.test_transform,download=True)
        # elif self.configs['dataset']=='celeba':
        #     train_data=torchvision.datasets.CelebA(dataset_path,split='train',target_type='identity',download=True,transform=self.train_transform)
        #     test_data=torchvision.datasets.CelebA(dataset_path,split='valid',target_type='identity',download=True,transform=self.test_transform)

        if self.configs['mode'] == 'eval': # eval 일때 dataset의 transform은 validation따라가기
            train_data.transform = copy.deepcopy(test_data.transform)
            train_data.train = True          
        return train_data, test_data

    def get_dataloader(self):
        train_data, test_data = self.get_dataset()
        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        
        if self.configs['ddp']:
            num_tasks=torch.distributed.get_world_size()
            train_sampler=torch.utils.data.distributed.DistributedSampler(train_data,num_replicas=num_tasks,rank=torch.distributed.get_rank(),shuffle=True)
            test_sampler=torch.utils.data.DistributedSampler(test_data,shuffle=False)
            train_shuffle=False
        else:
            if self.configs['labels_per_class'] is None:
                train_sampler=None
                test_sampler=None
                train_shuffle=True
            else: # specified labels per class
                # random sampler
                def get_sampler(labels, n=None):
                    # Only choose digits in self.configs['num_classes']
                    # n = number of labels per class for training
                    (indices, ) = np.where(reduce(__or__, [labels == i for i in np.arange(self.configs['num_classes'])]))
                    np.random.shuffle(indices)
                    indices_train = np.hstack([
                        list(filter(lambda idx: labels[idx] == i, indices))[:n]
                        for i in range(self.configs['num_classes'])
                    ])
                    indices_train = torch.from_numpy(indices_train)
                    sampler_train = SubsetRandomSampler(indices_train)
                    return sampler_train
                train_sampler=get_sampler(train_data.targets, self.configs['labels_per_class'])
                test_sampler=None
                train_shuffle=False
                torch.multiprocessing.set_sharing_strategy('file_system')
        train_data_loader = DataLoader(train_data, batch_size=self.configs['batch_size'],
                                    shuffle=train_shuffle, pin_memory=pin_memory,
                                    num_workers=self.configs['num_workers'], sampler=train_sampler
                                    )
        test_data_loader = DataLoader(test_data, batch_size=self.configs['batch_size'],
                                    shuffle=False, pin_memory=pin_memory,
                                    num_workers=self.configs['num_workers'],
                                    sampler=test_sampler
                                    )
        if self.configs['local_rank']==0:
            print("Using Datasets: ", self.configs['dataset'])
        return train_data_loader, test_data_loader
