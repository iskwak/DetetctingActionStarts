"""Hantman hungarian pytorch."""
from __future__ import print_function, division
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from collections import OrderedDict


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride, padding=1,
        bias=False
    )


class Hantman3DConv(nn.Module):
    # based on C3D model.
    def __init__(self):
        super(Hantman3DConv, self).__init__()
        self.convs = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu1', nn.ReLU(True)),
            ('maxpool1', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),

            ('conv2', nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu2', nn.ReLU(True)),
            ('maxpool2', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv3a', nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu3a', nn.ReLU(True)),
            ('conv3b', nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu3b', nn.ReLU(True)),
            ('maxpool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            # ('conv4a', nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            # ('relu4a', nn.ReLU(True)),
            # ('conv4b', nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            # ('relu4b', nn.ReLU(True)),
            # ('maxpool4', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv4a', nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu4a', nn.ReLU(True)),
            ('conv4b', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('relu4b', nn.ReLU(True)),
            ('maxpool4', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            # ('conv5a', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            # ('relu5a', nn.ReLU(True)),
            # ('conv5b', nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            # ('relu5b', nn.ReLU(True)),
            # ('maxpool5', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))
        ]))
        self.fc1 = nn.Linear(2 * 100352, 4096)
        # self.fc1 = nn.Linear(100352, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(4096, 7)
        self.sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        input1 = self.convs(inputs[0])
        input2 = self.convs(inputs[1])

        input1 = input1.view(input1.size(0), -1)
        input2 = input1.view(input2.size(0), -1)
        both = torch.cat([input1, input2], dim=1)

        features = self.fc1(both)
        x = self.relu(features)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.fc3(x)

        x = self.sigmoid(x)

        return x, features
