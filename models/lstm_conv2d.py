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

gflags.DEFINE_string(
    "hantman_perframeloss", "MSE", "Perframe loss")
gflags.DEFINE_float(
    "hantman_perframe_weight", 1.0, "Perframe weighting")
gflags.DEFINE_float(
    "hantman_struct_weight", 1.0, "Structured weighting")
gflags.DEFINE_float("hantman_tp", 2.0, "Weight for true positives.")
gflags.DEFINE_float("hantman_fp", 1.0, "Weight for false positives.")
gflags.DEFINE_float("hantman_fn", 20.0, "Weight for false negatives.")
# gflags.ADOPT_module_key_flags(lstm_flags)


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
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class HantmanHungarianImage(nn.Module):
    """Hantman mouse LSTM with hungarian matching."""
    def __init__(self, opts=None, hidden_dim=64,
                 output_dim=6, label_weight=None):
        super(HantmanHungarianImage, self).__init__()
        # if opts is not none, then use opts over provided parameters.
        input_dims = [1024]
        self.opts = opts
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.n_layers = 2
        # does batch size need to be a member variable?
        # self.batch_size = batch_size
        self.output_dim = output_dim
        self.label_weight = label_weight

        # network layers
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.seq = HantmanFeedForward()
        self.fc1 = nn.Linear(1024, hidden_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, self.n_layers)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # not sure if this is needed... but makes things easier later.
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size, use_cuda=False):
        """Initializes the hidden and memory cell with given batch size."""
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # The tuple reprensents hidden cell and memory cell.
        hidden = [
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False),
            autograd.Variable(
                torch.zeros(self.n_layers, batch_size, self.hidden_dim),
                requires_grad=False)
        ]
        # hidden = [
        #     autograd.Variable(torch.zeros(
        #         self.n_layers, batch_size, self.hidden_dim),
        #         requires_grad=False)
        #     for i in range(len(self.input_dims))
        # ]
        if use_cuda is not None and use_cuda >= 0:
            hidden = [init_hid.cuda() for init_hid in hidden]

        hidden = tuple(hidden)
        return hidden

    # def forward(self, side, front, hidden):
    def forward(self, inputs, hidden):
        """Forward pass for the network."""
        import pdb; pdb.set_trace()
        embed = self.seq(inputs)
        batch_size = embed.size(0)
        both = torch.cat(inputs, dim=2)

        embeds = both.view(-1, sum(self.input_dims))
        import pdb; pdb.set_trace()
        # embeds = self.bn1(embeds)
        embeds = self.fc1(embeds)
        # embeds = self.fc1(both.view(-1, sum(self.input_dims)))
        embeds = self.relu1(embeds)

        embeds = embeds.view(-1, batch_size, self.hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc2(lstm_out.view(-1, lstm_out.size(2)))
        out = out.view(-1, batch_size, self.output_dim)
        out = self.sigmoid(out)
        return out, hidden
