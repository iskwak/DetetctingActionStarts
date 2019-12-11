"""Based off of convert_hantman_data.py"""
from __future__ import print_function,division
import numpy
import argparse
import h5py
import scipy.io as sio
import helpers.paths as paths
# import helpers.git_helper as git_helper
import os
import time
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import PIL
import cv2
import cv
import sys

rng = numpy.random.RandomState(123)

# g_all_exp_dir = "/mnt/"
train_files = '/media/drive3/kwaki/data/mpiicooking2/experimentalSetup/sequencesTrainAttr.txt'
test_files = '/media/drive3/kwaki/data/mpiicooking2/experimentalSetup/sequencesTest.txt'

out_dir = "/media/drive3/kwaki/data/mpiicooking2/hdf5"


def process_file(vgg, preproc, filename):
    print("hi")


def main():
    files = []
    with open(train_files, "r") as train:
        for line in train:
            temp = line.strip() + '-cam-002.hdf5'
            files.append(temp)
    train_hdf5 = os.path.join(out_dir, "train.hdf5")
    with h5py.File(train_hdf5, "w") as train_data:
        train_data["exp_names"] = files
        train_data.create_group("exps")
        for exp in files:
            train_data["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/"
            )

    files = []
    with open(test_files, "r") as train:
        for line in train:
            temp = line.strip() + '-cam-002.hdf5'
            files.append(temp)
    train_hdf5 = os.path.join(out_dir, "test.hdf5")
    with h5py.File(train_hdf5, "w") as train_data:
        train_data["exp_names"] = files
        train_data.create_group("exps")
        for exp in files:
            train_data["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/"
            )


if __name__ == "__main__":
    # opts = setup_opts(opts)
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)

    main()
