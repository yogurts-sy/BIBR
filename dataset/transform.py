import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, IMG, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    IMG = ImageOps.expand(IMG, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    IMG = IMG.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, IMG, mask


def hflip(img, IMG, mask, p=0.5):
    if random.random() < p:
        # Image.FLIP_LEFT_RIGHT：左右水平翻转；
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        IMG = IMG.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, IMG, mask

def hflip3(img, mask, p=0.5):
    if random.random() < p:
        # Image.FLIP_LEFT_RIGHT：左右水平翻转；
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # common set
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def normalize2(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # common set
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, IMG, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    IMG = IMG.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, IMG, mask

def resize2(imgA, imgB, mask, A, B, M, ratio_range):
    w, h = imgA.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)
    
    ow = 128
    oh = 128

    imgA = imgA.resize((ow, oh), Image.BILINEAR)
    imgB = imgB.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)

    A = A.resize((ow, oh), Image.BILINEAR)
    B = B.resize((ow, oh), Image.BILINEAR)
    M = M.resize((ow, oh), Image.NEAREST)
    return imgA, imgB, mask, A, B, M

def resize3(imgA, imgB, mask, A, B, M, ratio_range):    
    w, h = imgA.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    imgA = imgA.resize((ow, oh), Image.BILINEAR)
    imgB = imgB.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)

    A = A.resize((ow, oh), Image.BILINEAR)
    B = B.resize((ow, oh), Image.BILINEAR)
    M = M.resize((ow, oh), Image.NEAREST)
    return imgA, imgB, mask, A, B, M


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

# cutmix框
def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_cutmix_box_for_aug(img_size, center_row, center_col, size_min=0.04, size_max=0.2, ratio_1=0.3, ratio_2=1/0.6):
    mask = torch.zeros(img_size, img_size)

    for i in range(3):
        size = np.random.uniform(size_min, size_max) * img_size * img_size  # 44 * 44 --> 114 * 114
        for j in range(5):
            ratio = np.random.uniform(ratio_1, ratio_2) 
            cutmix_h = int(np.sqrt(size * ratio))        
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h_half = int(cutmix_h / 2)
            cutmix_w_half = int(cutmix_w / 2)

            # 提升框的大小
            if center_row + cutmix_h_half > img_size:
                cutmix_h_half = img_size - center_row - 1
                cutmix_h_half = cutmix_h_half if center_row - cutmix_h_half >= 0 else center_row - 1
            
            elif center_row - cutmix_h_half < 0:
                cutmix_h_half = center_row - 1
                cutmix_h_half = cutmix_h_half if center_row + cutmix_h_half <= img_size else img_size - center_row - 1

            if center_col + cutmix_w_half > img_size:
                cutmix_w_half = img_size - cutmix_w_half - 1
                cutmix_w_half = cutmix_w_half if center_col - cutmix_w_half >= 0 else center_col - 1
            
            elif center_col - cutmix_w_half < 0:
                cutmix_w_half = center_col - 1
                cutmix_w_half = cutmix_w_half if center_col + cutmix_w_half <= img_size else img_size - center_col - 1

            if (center_row + cutmix_h_half <= img_size) and (center_row - cutmix_h_half >= 0) and \
                (center_col + cutmix_w_half <= img_size) and (center_col - cutmix_w_half >= 0):
                mask[center_row-cutmix_h_half:center_row+cutmix_h_half, center_col-cutmix_w_half:center_col+cutmix_w_half] = 1
                return mask
    return mask
