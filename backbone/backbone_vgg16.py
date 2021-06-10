"""
@File  : backbone_vgg16.py
@Author: CodeCat
@Time  : 2021/6/8 17:15
"""
import torch
import torch.nn as nn
from collections import OrderedDict

from .vgg16_model import vgg16


class SSDFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(SSDFeatureExtractor, self).__init__()

        _, _, maxpool3_pos, maxpool4_pos, _ = (i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d))

        # 对于maxpool3的ceil_mode设置为True，以满足输出图像的尺寸为38x38
        backbone[maxpool3_pos].ceil_mode = True

        self.features = nn.Sequential(*backbone[:maxpool4_pos])

        extra = nn.ModuleList([
            # conv8_2
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            # conv9_2
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv10_2
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            # conv11_2
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        ])

        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
            # FC6
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # FC7
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        extra.insert(0, nn.Sequential(
            # maxpool4一直到conv5_3，略过maxpool5
            *backbone[maxpool4_pos:-1],
            fc,
        ))

        self.extra = extra
        self.out_channels = [512, 1024, 512, 256, 256, 256]

        # init_weight
        for layer in self.extra.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.features(x)
        output = [x]
        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def vgg16_backbone(pretrain_path='', trainable_layers=3):
    backbone = vgg16().features
    if pretrain_path != "":
        backbone.load_state_dict(torch.load(pretrain_path), strict=False)

    stage_indices = [i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d)]
    num_stages = len(stage_indices)

    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for layer in backbone[:freeze_before]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)

    return SSDFeatureExtractor(backbone)