import tarfile
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import os

import os
import torch
import os.path
import torch.utils.data as data
from scipy.io import loadmat

class Flower102(data.Dataset):
    def __init__(self, root_dir, label_file='imagelabels.mat', mode = 'train', data_split='setid.mat', transform=None):
        self.all_labels = loadmat(os.path.join(root_dir,label_file))['labels'][0]
        imglabel_map = os.path.join(root_dir, 'imagelabels.mat')
        setid_map = os.path.join(root_dir, 'setid.mat')
        assert os.path.exists(imglabel_map), 'Mapping txt is missing ({})'.format(imglabel_map)
        assert os.path.exists(setid_map), 'Mapping txt is missing ({})'.format(setid_map)
        if not os.path.exists(os.path.join(root_dir,'jpg')):
            for filename in ['102flowers.tgz']:# '102segmentations.tgz',
                with tarfile.open(os.path.join(root_dir,filename), "r:gz") as tar:
                    tar.extractall(path=root_dir)
        mapping = {'train':'trnid','val':'valid','test':'tstid'}        
        self.split = loadmat(os.path.join(root_dir,data_split))[mapping[mode]][0]        
        self.root = os.path.join(root_dir,'jpg')        
        self.images = [os.path.join(self.root,'image_{:05d}.jpg'.format(id))for id in self.split]        
        self.labels = torch.tensor([self.all_labels[id-1]-1 for id in self.split]        ,dtype=torch.long)
        self.transforms = transform 
        
    def __len__(self):        
        return len(self.labels)    

    def __getitem__(self, index):
        """        Args:            index (int): Index
        Returns:            tuple: (image, target) where target is index of the target class.
        """        
        img = Image.open(self.images[index]).convert('RGB')  
        target = self.labels[index]        
        if self.transforms:            
            img = self.transforms(img)        
        return img, target
