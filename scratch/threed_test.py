"""Test 3d convolutions."""
from __future__ import print_function,division
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
from helpers.hantman_sampler import HantmanSeqFrameSampler
import numpy
import h5py
from collections import OrderedDict
import gc
import time
from models.hantman_3dconv import Hantman3DConv

# class Net(nn.Module):
#     def __init__(self):
#         self.layer1 = nn.Conv3d()

# layer = nn.Conv3d(3, 64, 3, stride=2, padding=1)
# conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
# bn1 = nn.BatchNorm3d(64)
# relu = nn.ReLU(inplace=True)
# maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
# layer1 = self._make_layer(block, 64, layers[0])
# layer2 = self._make_layer(block, 128, layers[1], stride=2)
# layer3 = self._make_layer(block, 256, layers[2], stride=2)
# layer4 = self._make_layer(block, 512, layers[3], stride=2)
# avgpool = nn.AvgPool2d(7, stride=1)
# fc = nn.Linear(2 * 512 * block.expansion, 1000)
# seq_net = nn.Sequential(OrderedDict([
#     # ('conv1', nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)),
#     ('conv1', nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)),
#     ('bn1', nn.BatchNorm3d(64)),
#     ('relu1', nn.ReLU(inplace=True)),
#     ('maxpool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#     ('conv2', nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)),
#     ('bn2', nn.BatchNorm3d(128)),
#     ('relu2', nn.ReLU(inplace=True)),
#     ('maxpool2', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#     ('conv3', nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)),
#     ('bn3', nn.BatchNorm3d(256)),
#     ('relu3', nn.ReLU(inplace=True)),
#     ('maxpool3', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),

#     ('conv4', nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)),
#     ('bn4', nn.BatchNorm3d(256)),
#     ('relu4', nn.ReLU(inplace=True)),
#     ('maxpool4', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
#     # ('conv4', nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)),
#     # ('bn4', nn.BatchNorm3d(512)),
#     # ('relu4', nn.ReLU(inplace=True)),
#     # ('maxpool4', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
# ]))

# fc_net = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(2 * 50176, 4096)),
#     # ('fc1', nn.Linear(2 * 100352, 4096)),
#     ('relu1', nn.ReLU(True)),
#     ('drop1', nn.Dropout()),
#     ('fc2', nn.Linear(4096, 6)),
#     ('sigmoid', nn.Sigmoid())
# ]))

# seq_net = seq_net.cuda()
# fc_net = fc_net.cuda()
# import pdb; pdb.set_trace()
batch_size = 8
num_frames = 10
# input = autograd.Variable(torch.randn(batch_size, 3, num_frames, 224, 224))
moo = Hantman3DConv()
moo = moo.cuda()

rng = numpy.random.RandomState(123)
# train_file = "/media/drive2/kwaki/data/hantman_processed/20170827_vgg/one_mouse_one_day_train.hdf5"
train_file = "/nrs/branson/kwaki/data/20170827_vgg/one_mouse_multi_day_train.hdf5"
frame_path = "/media/drive1/data/hantman_frames"
with h5py.File(train_file, "r") as train_data:
    sampler = HantmanSeqFrameSampler(rng, train_data, frame_path,
                                     batch_size, use_pool=False)
    tic = time.time()
    for i in range(sampler.num_batch):
        print(i)
        cow = sampler.get_minibatch()
        cow[0] = cow[0].cuda()
        cow[1] = cow[1].cuda()
        cow[2] = cow[2].cuda()
        # img1 = seq_net(cow[0])
        # img1 = img1.view(img1.size(0), -1)
        # img2 = seq_net(cow[1])
        # img2 = img1.view(img2.size(0), -1)

        # both = torch.cat([img1, img2], dim=1)

        # out = fc_net(both)
        moo(cow[:2])
        import pdb; pdb.set_trace()
        # del img1
        # del img2
        # import pdb; pdb.set_trace()
        # cow = []
        # gc.collect()

    # import pdb; pdb.set_trace()
    print("moo")
print(time.time() - tic)


# seq_net = nn.Sequential(OrderedDict([
#     ('conv1', nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)),
#     ('bn1', nn.BatchNorm3d(64)),
#     ('relu1', nn.ReLU(inplace=True)),
#     ('maxpool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#     ('conv2', nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)),
#     ('bn2', nn.BatchNorm3d(128)),
#     ('relu2', nn.ReLU(inplace=True)),
#     ('maxpool2', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#     ('conv3', nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)),
#     ('bn3', nn.BatchNorm3d(256)),
#     ('relu3', nn.ReLU(inplace=True)),
#     ('maxpool3', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
#     ('conv4', nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)),
#     ('bn4', nn.BatchNorm3d(512)),
#     ('relu4', nn.ReLU(inplace=True)),
#     ('maxpool4', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
# ]))
# fc_net = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(2 * 100352, 4096)),
#     ('relu1', nn.ReLU(True)),
#     ('drop1', nn.Dropout()),
#     ('fc2', nn.Linear(4096, 6)),
#     ('sigmoid', nn.Sigmoid())
# ]))