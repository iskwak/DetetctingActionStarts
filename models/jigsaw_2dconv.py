"""Resnet with different output layer."""
from __future__ import print_function, division
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from collections import OrderedDict
import torchvision.models.resnet as resnet


# class Jigsaw2DConv(nn.Module):
#     # based on Resnet50 in models. Mainly need to change the fc layer
#     # and the forward pass.
#     def __init__(self):
#         super(Jigsaw2DConv, self).__init__()
#         model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3])
#         model.fc = nn.Linear(2048, 10)
#         self.model = model

#     def forward(self, image):
#         return self.model(image)

def jigsaw_2dconv(num_classes, **kwargs):
    """Create a model with different output layer."""
    # model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
    model = resnet.resnet34()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
