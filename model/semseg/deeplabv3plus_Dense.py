import model.backbone.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F

class DeepLabV3Plus_Dense(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus_Dense, self).__init__()

        # backbone设置
        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                         replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        c1_channels = 256
        c2_channels = 512
        c3_channels = 1024
        c4_channels = 2048

        # ASPP特征提取模块
		# 利用不同膨胀率的膨胀卷积进行特征提取
        # self.head = ASPPModule(high_channels, cfg['dilations'])
        self.head = DenseASPP(c4_channels, 512, 256)

        # 浅层特征边
        self.reduce = nn.Sequential(nn.Conv2d(c1_channels + c2_channels + c3_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(c4_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x1, x2=None, need_fp=False, mode='cd'):  # x1, x2 [4, 3, 256, 256]
        h, w = x1.shape[-2:]

        if mode == 'cd':
            feats1 = self.backbone.base_forward(x1)
            c11, c12, c13, c14 = feats1[0], feats1[1], feats1[2], feats1[3] # 256  -  512  -  1024  -  2048

            feats2 = self.backbone.base_forward(x2)
            c21, c22, c23, c24 = feats2[0], feats2[1], feats2[2], feats2[3] # 256  -  512  -  1024  -  2048

            c1 = (c11 - c21).abs()  # [4, 256, 64, 64]    浅层特征
            c2 = (c12 - c22).abs()
            c3 = (c13 - c23).abs()
            c4 = (c14 - c24).abs()  # [4, 2048, 16, 16]   加强特征
        else:
            feats1 = self.backbone.base_forward(x1)
            c1, c2, c3, c4 = feats1[0], feats1[1], feats1[2], feats1[3]     # 256  -  512  -  1024  -  2048

        
        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c2, nn.Dropout2d(0.5)(c2))),
                                torch.cat((c3, nn.Dropout2d(0.5)(c3))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        out = self._decode(c1, c2, c3, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c2, c3, c4):
        h, w = c1.shape[-2:] # 64, 64
        c2 = F.interpolate(c2, size=(h, w), mode="bilinear", align_corners=True)
        c3 = F.interpolate(c3, size=(h, w), mode="bilinear", align_corners=True)
        c1 = torch.cat([c1, c2, c3], dim=1)  # 256 + 512 + 1024

        c1 = self.reduce(c1) # [8, 48, 64, 64]

        c4 = self.head(c4)   # [8, 256, 16, 16]
        c4 = F.interpolate(c4, size=(h, w), mode="bilinear", align_corners=True)

        feature = torch.cat([c4, c1], dim=1) # [8, 256 + 48, 64, 64]
        feature = self.fuse(feature)         # [8, 256, 64, 64]

        out = self.classifier(feature)

        return out


class _StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(_StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)  # 将输出的特征与原始输入特征结合


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class DenseASPP(nn.Module):
    # 2048, 512, 256
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DenseASPP, self).__init__()
        # 2048, 512, 256
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        # 2048 + 256, 512, 256
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        # 2048 + 256 * 2, 512, 256
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        # 2048 + 256 * 3, 512, 256
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        # 2048 + 256 * 4, 512, 256
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        self.SP = _StripPooling(2048, up_kwargs={'mode': 'bilinear', 'align_corners': True})

        self.project = nn.Sequential(nn.Conv2d(in_channels * 2 + inter_channels2 * 5, inter_channels2, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels2),
                                     nn.ReLU(True))

    def forward(self, x):

        x1 = self.SP(x)                     # [b, 2048, 16, 16]
        aspp3 = self.aspp_3(x)              # [b, 256, 16, 16]

        x = torch.cat([aspp3, x], dim=1)    # [b, 2048 + 256, 16, 16]

        aspp6 = self.aspp_6(x)              # [b, 256, 16, 16]
        x = torch.cat([aspp6, x], dim=1)    # [b, 2048 + 256 * 2, 16, 16]

        aspp12 = self.aspp_12(x)            # [b, 256, 16, 16]
        x = torch.cat([aspp12, x], dim=1)   # [b, 2048 + 256 * 3, 16, 16]

        aspp18 = self.aspp_18(x)            # [b, 256, 16, 16]
        x = torch.cat([aspp18, x], dim=1)   # [b, 2048 + 256 * 4, 16, 16]

        aspp24 = self.aspp_24(x)            # [b, 256, 16, 16]
        x = torch.cat([aspp24, x], dim=1)   # [b, 2048 + 256 * 5, 16, 16]
        x = torch.cat([x, x1], dim=1)       # [b, 2048 * 2 + 256 * 5, 16, 16]

        return self.project(x)
