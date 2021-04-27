import numpy as np
import sys
import os
import random
import torch

def seed_everything(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dice(pred, label):
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)))
    return dice_val