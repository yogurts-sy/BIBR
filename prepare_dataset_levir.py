import argparse
import logging
import os

os.environ['LOCAL_RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["RANK"] = '0'
os.environ["WORLD_SIZE"] = '1'
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8096'

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import cv2

from dataset.semicd import SemiCDDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
import matplotlib.pyplot as plt
plt.set_cmap('gray')

from mmcv.ops.nms import nms
from util.visualize import make_numpy_grid, visualize_eval, visualize_patches
ops = 'mmcv'

# Prepare BDN dataset for LEVIR-CD 40%

config_path = "/data/suyou/Codes/BIBR/configs/levir.yaml"
labeled_id_path = "/data/suyou/Codes/BIBR/splits/levir/40%/labeled.txt"
cd_save_path = "/data/suyou/Codes/BIBR/exp/levir/pretrained"                         # Pretrained CD model for Contour Extract and Match

# Training dataset save path
txt_save_path = "/data/suyou/Codes/BIBR"
patches_save_path = "/data/suyou/Codes/datasets/BIBR-LEVIR/train/patches"
label_save_path = "/data/suyou/Codes/datasets/BIBR-LEVIR/train/label"
local_rank = 0
port = "8090"

os.makedirs(patches_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)


def compute_iou(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    try:
        iou = intersection_area / union_area
    except ZeroDivisionError as e:
        return 0
    return iou

def filter_small(mask):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output1 = np.zeros((256, 256))

    for i, (contour1) in enumerate(contours):
        area = cv2.contourArea(contour1)
        if area < 30:
            continue

        cv2.drawContours(output1, [contour1], -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((1, 1), np.uint8)
    output1 = cv2.dilate(output1, kernel, iterations = 1)

    return output1

def filter_inside(mask, dets):
    index = []
    inside_index = []
    ious = []

    dets_indices = dets.astype(int)[:, :4]
    for i, (x1, y1, x2, y2) in enumerate(dets_indices):
        # mask1 = np.zeros((256, 256))
        mask1 = mask[y1:y2, x1:x2]
        ratio = cv2.countNonZero(mask1) / 2304
        if ratio < 0.90:
            index.append(i)
        else:
            inside_index.append(i)
        ious.append(ratio)
    
    sdets = dets[index]
    inside_sdets = dets[inside_index]

    return sdets# , inside_sdets, ious

def filter_AB(intersection, dets):
    index = []

    dets_indices = dets.astype(int)[:, :4]
    for pid, (x1, y1, x2, y2) in enumerate(dets_indices):
        intersection_mask = intersection[y1:y2, x1:x2]

        area = cv2.countNonZero(intersection_mask)

        if area == 0:
            index.append(pid)

    sdets = dets[index]

    return sdets


def compute_iou1(tensor1, tensor2):
    mask1 = tensor1.cpu().numpy().astype(int)
    mask2 = tensor2.cpu().numpy().astype(int)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)

    try:
        iou = intersection_area / union_area
    except ZeroDivisionError as e:
        return 0
    return iou

def compute_iou2(contour1, contour2):
    mask1 = np.zeros((256, 256))
    mask2 = np.zeros((256, 256))

    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)

    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)

    intersection_area = cv2.countNonZero(intersection)
    union_area = cv2.countNonZero(union)

    try:
        iou = intersection_area / union_area
    except ZeroDivisionError as e:
        return 0
    return iou



def find_float_boundary(maskdt, width=3):
    # Extract boundary from instance mask
    maskdt = torch.Tensor(maskdt).unsqueeze(0).unsqueeze(0)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt.permute(1, 0, 2, 3), boundary_finder,
                            stride=1, padding=width//2).permute(1, 0, 2, 3)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask[0, 0].numpy()


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
    sdets = sdets.copy()
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets

def get_dets(maskdt, patch_size, iou_thresh=0.3):
    """Generate patch proposals from the coarse mask.

    Args:
        maskdt (array): H, W
        patch_size (int): [description]
        iou_thresh (float, optional): useful for nms. Defaults to 0.3.

    Returns:
        array: filtered bboxs. shape N, 4. each row contain x1, y1, 
            x2, y2, score. e.g.
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
    """
    fbmask = find_float_boundary(maskdt)
    ys, xs = np.where(fbmask)
    scores = fbmask[ys, xs]
    dets = np.stack([xs-patch_size//2, ys-patch_size//2,
                     xs+patch_size//2, ys+patch_size//2, scores]).T
    if ops == 'mmdet':
        _, inds = nms(dets, iou_thresh)
    else:
        _, inds = nms(np.ascontiguousarray(dets[:, :4], np.float32),
                      np.ascontiguousarray(dets[:, 4], np.float32),
                      iou_thresh)
    # 经过nms过滤后的bbox
    sdets = dets[inds]

    H, W = maskdt.shape
    return _force_move_back(sdets, H, W, patch_size) # , _force_move_back(dets, H, W, patch_size)

def crop(img, maskdt, maskgt, dets):
    dets = dets.astype(int)[:, :4]
    # crop
    img_patches, dt_patches, gt_patches = [], [], []
    for x1, y1, x2, y2 in dets:
        img_patches.append(img[y1:y2, x1:x2, :])
        dt_patches.append(maskdt[y1:y2, x1:x2])
        gt_patches.append(maskgt[y1:y2, x1:x2])
    return img_patches, dt_patches, gt_patches

def filter(filter_mode):
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=port)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_zoo = {'deeplabv3plus': DeepLabV3Plus}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    trainset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], labeled_id_path)
   
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)

    valset = SemiCDDataset(cfg['dataset'], cfg['data_root'], 'val')
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    
    # loading change detection model
    if os.path.exists(os.path.join(cd_save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(cd_save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # just for visualization
    if filter_mode == 'train':
        dataLoader = trainloader
        base_path = '/data/suyou/Codes/BIBR/visualize/train'
        save_txt = 'train.txt'
    else:
        dataLoader = valloader
        base_path = '/data/suyou/Codes/BIBR/visualize/test'
        save_txt = 'test.txt'

    for i, (imgA1, imgB1, _, _, mask, A, B, _, id) in enumerate(dataLoader):

        imgA1, imgB1, mask = imgA1.cuda(), imgB1.cuda(), mask.cuda()

        pred = model(imgA1, imgB1)
        outputs = pred.argmax(dim=1)

        intersection, union, _ = intersectionAndUnion(outputs.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'], 255)

        iou = intersection / (union + 1e-10)

        if iou[1] > 0.5:   # changed iou > 0.5

            id = id[0].split(".")[0]
            # os.makedirs(os.path.join(base_path, id), exist_ok=True)
            # visualize_save_path = os.path.join(base_path, id)
            # visualize_eval(A, B, pred, mask, visualize_save_path, f'{id}')

            print(id)
            pred_A = model(imgA1, mode='seg')
            pred_B = model(imgB1, mode='seg')

            gt_mask = mask.detach().squeeze(0).cpu().numpy()
            gt_mask = (gt_mask * 255).astype(np.uint8)          # (256, 256)
            # cv2.imwrite(f'{visualize_save_path}/{id}_gt_mask.png', gt_mask)
            gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pred_A = torch.argmax(pred_A[0].detach(), dim=0, keepdim=True).squeeze(0)
            pred_A_mask = pred_A.detach().cpu().numpy()
            pred_A_mask = (pred_A_mask * 255).astype(np.uint8)  # (256, 256)
            # cv2.imwrite(f'{visualize_save_path}/{id}_predA_mask.png', pred_A_mask)
            pred_A_contours, _ = cv2.findContours(pred_A_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pred_B = torch.argmax(pred_B[0].detach(), dim=0, keepdim=True).squeeze(0)
            pred_B_mask = pred_B.detach().cpu().numpy()
            pred_B_mask = (pred_B_mask * 255).astype(np.uint8)  # (256, 256)
            # cv2.imwrite(f'{visualize_save_path}/{id}_predB_mask.png', pred_B_mask)
            pred_B_contours, _ = cv2.findContours(pred_B_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            AB_intersection = cv2.bitwise_and(pred_A_mask, pred_B_mask)

            AB_intersection[gt_mask == 255] = 0
            AB_intersection = filter_small(AB_intersection)

            # cv2.imwrite(f'{visualize_save_path}/{id}_AB_intersection_filtered.png', AB_intersection)

            matched_contours = []

            for contour in gt_contours:
                iou_best = 0
                matched_contour = None
                for contour_A in pred_A_contours:
                    iou_A = compute_iou2(contour, contour_A)
                    if iou_A > iou_best:
                        matched_contour = (contour, contour_A, 'A')
                        iou_best = iou_A
                for contour_B in pred_B_contours:
                    iou_B = compute_iou2(contour, contour_B)
                    if iou_B > iou_best:
                        matched_contour = (contour, contour_B, 'B')
                        iou_best = iou_B
                if matched_contour:
                    matched_contours.append(matched_contour)
            
            for ii, (contour_mask, _, from_where) in enumerate(matched_contours):
                mask1 = np.zeros_like(gt_mask)
                cv2.drawContours(mask1, [contour_mask], -1, 1, cv2.FILLED)

                mask2 = np.zeros_like(gt_mask)
                cv2.drawContours(mask2, [contour_mask], -1, (255, 255, 255), cv2.FILLED)
                # cv2.imwrite(f'{visualize_save_path}/{id}_contours_{ii}_match_{from_where}.png', mask2)

                dets_one = get_dets(mask1, 48, 0.3)

                dets_one = filter_inside(mask1, dets_one)

                # filter building in both A and B
                dets_one = filter_AB(AB_intersection, dets_one)

                if len(dets_one) == 0:
                    continue

                img = A if from_where == 'A' else B
                img = make_numpy_grid(img[0])
                maskgt = mask.squeeze(0).detach().cpu().numpy()
                img_one_patches, _, gt_one_patches = crop(img, mask1, maskgt, dets_one)

                with open(os.path.join(txt_save_path, save_txt), 'a') as txt_file:
                    for iii in range(len(img_one_patches)):
                        txt_file.write(f"{id}_contours_{ii}_patch_{iii}.png" + '\n')
                        
                        # saving contours
                        plt.imsave(f'{patches_save_path}/{id}_contours_{ii}_patch_{iii}.png', img_one_patches[iii])
                        plt.imsave(f'{label_save_path}/{id}_contours_{ii}_patch_{iii}.png', gt_one_patches[iii] * 255)

                        # visualize_patches(img_one_patches[iii], gt_one_patches[iii], visualize_save_path, f"{id}_contours_{ii}_patch_{iii}")


def main():
    # Step.1 Prepare contour dataset for train set
    filter_mode = 'train'

    # Step.2 Prepare contour dataset for test set
    # filter_mode = 'test'

    filter(filter_mode)

if __name__ == "__main__":
    # Prepare contour dataset for LEVIR-CD
    main()