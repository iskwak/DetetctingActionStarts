"""Hantman hungarian pytorch."""
from __future__ import print_function, division
import torch
import torch.autograd as autograd
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import gflags
import numpy as np
# import flags.lstm_flags as lstm_flags
import torchvision.models as models
import torchvision.models.resnet as resnet
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
import math


class HantmanFeedForward(nn.Module):
    # based on Resnet18 code in models. Mainly need to change the fc layer
    # and the forward pass.
    def __init__(self, pretrained=False):
        super(HantmanFeedForward, self).__init__()
        self.make_base_model(pretrained)

    def make_base_model(self, pretrained):
        block = resnet.BasicBlock
        layers = [2, 2, 2, 2]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2 * 512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # fill up the parameters.
        if pretrained is True:
            self.load_state_dict(
                model_zoo.load_url(resnet.model_urls['resnet18']))
        # after loading up the pretrained weights. Tweak the model.
        # Mainly... the final FC layer needs to be 2 times the width.
        self.fc = nn.Linear(2 * 512 * block.expansion, 2 * 512 * block.expansion)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(2 * 512 * block.expansion, 2 * 512 * block.expansion)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()
        self.fc_class = nn.Linear(2 * 512 * block.expansion, 7)
        self.sigmoid = nn.LogSoftmax(dim=1)
        # self.fc_class = nn.Linear(2 * 512 * block.expansion, 6)
        # self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def process_image(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, inputs):
        input1 = self.process_image(inputs[0])
        input2 = self.process_image(inputs[1])
        both = torch.cat([input1, input2], dim=1)

        x = self.fc(both)

        # features = self.relu1(x)
        # x = self.dropout1(features)
        x = self.relu1(x)
        x = self.dropout1(x)

        # x = self.fc2(x)
        # x = self.relu2(x)
        features = self.fc2(x)
        x = self.relu2(features)

        x = self.dropout2(x)
        x = self.fc_class(x)
        x = self.sigmoid(x)

        return x, features


class HantmanFeedForwardVGG(nn.Module):

    def __init__(self, pretrained=False):
        super(HantmanFeedForwardVGG, self).__init__()
        num_classes = 1000
        self.make_model(vgg.make_layers(vgg.cfg['D']))
        if pretrained is True:
            self.load_state_dict(
                model_zoo.load_url(vgg.model_urls['vgg16']))
        # change up the classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * 512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7),
        )

    def make_model(self, features):
        num_classes = 1000
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, inputs):
        input1 = self.features(inputs[0])
        input1 = input1.view(input1.size(0), -1)

        input2 = self.features(inputs[1])
        input2 = input2.view(input2.size(0), -1)

        both = torch.cat([input1, input2], dim=1)
        x = self.classifier(both)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# # creating the hantman feedforward pretrained deck requires some surgery.
# def create_network(arch="resnet", pretrained=True):
#     """Helper function to create the pretrained network."""
