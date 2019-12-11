"""Hantman hungarian pytorch."""
from __future__ import print_function, division
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from collections import OrderedDict


class Jigsaw3DConv(nn.Module):
    # based on Resnet18 code in models. Mainly need to change the fc layer
    # and the forward pass.
    def __init__(self):
        super(Jigsaw3DConv, self).__init__()
        self.convs = nn.Sequential(OrderedDict([
            # ('conv1', nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)),
            ('conv1', nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=2, padding=1)),
            ('bn1', nn.BatchNorm3d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)),
            ('bn2', nn.BatchNorm3d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ('conv3', nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)),
            ('bn3', nn.BatchNorm3d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.fcs = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(4096, 4096)),
            ('relu1', nn.ReLU(True)),
            ('drop1', nn.Dropout()),
            # 16 cause there's the background label
            ('fc3', nn.Linear(4096, 16)),
            ('sigmoid', nn.LogSoftmax())
        ]))

    def forward(self, inputs):
        x = self.convs(inputs)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
