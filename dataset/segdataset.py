from dataset.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
from torch.utils.data import Dataset

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



class SegDataset(Dataset):
    def __init__(self, root, mode, size, id_path=None, val_id_path=None, nsample=None):
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            self.root = self.root + '/train'
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if (mode == 'train_l') and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            self.root = self.root + '/test'
            with open(val_id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        img = Image.open(os.path.join(self.root, 'patches', id)).convert('RGB')

        Totensor = transforms.ToTensor()
        IMG = Image.open(os.path.join(self.root, 'patches', id)).convert('RGB')

        mask = np.array(Image.open(os.path.join(self.root, 'label', id)).convert('L'))
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        ow = 140
        oh = 140

        # ow = 128
        # oh = 128

        img = img.resize((ow, oh), Image.BILINEAR)
        IMG = IMG.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id, Totensor(IMG)

        img, IMG, mask = resize(img, IMG, mask, (0.8, 1.2))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, IMG, mask = crop(img, IMG, mask, 140, ignore_value)
        # img, IMG, mask = crop(img, IMG, mask, 128, ignore_value)
        img, IMG, mask = hflip(img, IMG, mask)

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)

            return img, mask, id, Totensor(IMG)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2, id, Totensor(IMG)

    def __len__(self):
        return len(self.ids)
