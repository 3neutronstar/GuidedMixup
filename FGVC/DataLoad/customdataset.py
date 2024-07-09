from torch.utils.data.dataset import Dataset
import numpy as np
import copy
from PIL import Image
class IndexDataset(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self.dataset)

class TransformTensorDataset(Dataset):
    def __init__(self,dataset,transform):
        self.dataset=dataset
        self.transform=transform
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        data=self.transform(data)
        return data, target

    def __len__(self):
        return len(self.dataset)

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

class AddressBasedImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images # address of image
        self.y = labels # image label
        self.transforms = transforms # image transform
         
    def __len__(self):
        return (len(self.X)) # length of dataset
    
    def __getitem__(self, i):
        data = Image.open(self.X[i]) # open image
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data