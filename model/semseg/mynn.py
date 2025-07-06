"""
Custom Norm wrappers to enable sync BN, regular BN and for weight
initialization
"""
import torch
import torch.nn as nn


align_corners = True



def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=align_corners)


def Upsample2(x):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                     align_corners=align_corners)


def Down2x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=0.5, mode='bilinear', align_corners=align_corners)


def Up15x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=1.5, mode='bilinear', align_corners=align_corners)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=align_corners)
    return x_scaled


def DownX(x, scale_factor):
    '''
    scale x to the same size as y
    '''
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=align_corners, recompute_scale_factor=True)
    return x_scaled


def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=align_corners, recompute_scale_factor=True)
    return x_scaled
