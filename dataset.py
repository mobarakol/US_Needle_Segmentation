#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as standard_transforms


class USNeedleDataset(Dataset):
    def __init__(self, img_dir, is_train=False):
        self.is_train = is_train
        if self.is_train:
            self.img_anno_pairs = glob(os.path.join(img_dir, 'train/**_mask.png'))
        else:
            self.img_anno_pairs = glob(os.path.join(img_dir, 'test/**_mask.png'))

    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _target = Image.open(self.img_anno_pairs[index]).convert('L')
        _img = Image.open(self.img_anno_pairs[index][:-9] +'.png').convert('RGB')
        
        if self.is_train:
            if random.random() < 0.5:
                _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                _target = _target.transpose(Image.FLIP_LEFT_RIGHT)

        _img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
        _target = np.array(_target)
        _target[_target == 255] = 1
        _target = torch.from_numpy(np.array(_target)).long()
        return _img, _target