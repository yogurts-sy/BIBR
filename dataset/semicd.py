from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiCDDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u' or mode == 'train_ls' or mode == 'train_aug':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if (mode == 'train_l' or mode == 'train_ls' or mode == 'train_aug') and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]

        else:
            if name == 'dsifn':
                val_name = 'val.txt'
            else:
                val_name = 'test.txt'
            val_path = f'splits/{name}/{val_name}'
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        imgA = Image.open(os.path.join(self.root, 'A', id)).convert('RGB')
        imgB = Image.open(os.path.join(self.root, 'B', id)).convert('RGB')

        Totensor = transforms.ToTensor()
        # A = Totensor(Image.open(os.path.join(self.root, 'A', id)))
        # B = Totensor(Image.open(os.path.join(self.root, 'B', id)))
        # M = Totensor(Image.open(os.path.join(self.root, 'label', id)))
        A = Image.open(os.path.join(self.root, 'A', id))
        B = Image.open(os.path.join(self.root, 'B', id))
        M = Image.open(os.path.join(self.root, 'label', id))
        
        mask = np.array(Image.open(os.path.join(self.root, 'label', id)))
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val':
            imgA1, _ = normalize(imgA, mask)
            imgB1 = normalize(imgB)

            imgA2, mask2 = normalize2(imgA, mask)
            imgB2 = normalize2(imgB)
            return imgA1, imgB1, imgA2, imgB2, mask2, Totensor(A), Totensor(B), Totensor(M), id
        
        # ——————————————————弱增强——————————————————
        # imgA, imgB, mask = resize(imgA, imgB, mask, (0.8, 1.2))
        # imgA, imgB, mask, A, B, M = resize3(imgA, imgB, mask, A, B, M)
        # imgA, imgB, mask, A, B, M = crop2(imgA, imgB, mask, self.size, A, B, M)
        # imgA, imgB, mask, A, B, M = hflip2(imgA, imgB, mask, A, B, M)

        if self.mode == 'train_l':
            imgA1, _ = normalize(imgA, mask)
            imgB1 = normalize(imgB)

            imgA2, mask2 = normalize2(imgA, mask)
            imgB2 = normalize2(imgB)
            return imgA1, imgB1, imgA2, imgB2, mask2, Totensor(A), Totensor(B), Totensor(M), id

        if self.mode == 'train_aug':
            imgA = normalize(imgA)
            imgB = normalize(imgB)

            ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
            ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
            mask = torch.from_numpy(np.array(mask)).long()
            ignore_mask[mask == 255] = 255
            return imgA, imgB, mask, ignore_mask

        # ——————————————————强增强——————————————————
        # 仅针对于无标签图像
        imgA_w, imgB_w = deepcopy(imgA), deepcopy(imgB)
        imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA)
        imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB)

        # strong perturbations
        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s1)
        imgA_s1 = transforms.RandomGrayscale(p=0.2)(imgA_s1)
        imgA_s1 = blur(imgA_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(imgA_s1.size[0], p=0.5)

        if random.random() < 0.8:
            imgB_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s1)
        imgB_s1 = transforms.RandomGrayscale(p=0.2)(imgB_s1)
        imgB_s1 = blur(imgB_s1, p=0.5)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s2)
        imgA_s2 = transforms.RandomGrayscale(p=0.2)(imgA_s2)
        imgA_s2 = blur(imgA_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(imgA_s2.size[0], p=0.5)

        if random.random() < 0.8:
            imgB_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s2)
        imgB_s2 = transforms.RandomGrayscale(p=0.2)(imgB_s2)
        imgB_s2 = blur(imgB_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 255] = 255

        if self.mode == 'train_ls':
            return normalize(imgA_s1), normalize(imgB_s1), mask, ignore_mask, cutmix_box1

        # ignore_mask为无标签图像的标签
        return normalize(imgA_w), normalize(imgB_w), normalize(imgA_s1), normalize(imgB_s1), \
            normalize(imgA_s2), normalize(imgB_s2), ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
