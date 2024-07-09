
import torch
import PIL
from PIL import Image
import os
import pandas as pd
from scipy.io import loadmat
'''
Stanford Cars 196
'''
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [int(anno[i][-2][0, 0]) for i in range(anno.shape[0])]
    return results


class Cars196Dataset(torch.utils.data.Dataset):

    def __init__(self, root='Stanford_Cars', transform=None, target_transform=None, train=False, loader=pil_loader):

        img_folder = root
        pd_train = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
        pd_test = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat'), 'cars_test'))
        
        data = pd.concat([pd_train, pd_test])
        data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = torch.tensor(item['label'],dtype=torch.long)

        img = self.loader(os.path.join(self.root, file_path))
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
