import h5py
import os
import numpy

dataname = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5"

train_num_frames = []
with h5py.File(dataname, "r") as h5data:
    exp_names = h5data["exp_names"]

    for i in range(len(exp_names)):
        train_num_frames.append(
            h5data["exps"][exp_names[i]]["labels"].shape[0]
        )

dataname = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/test.hdf5"
test_num_frames = []
with h5py.File(dataname, "r") as h5data:
    exp_names = h5data["exp_names"]

    for i in range(len(exp_names)):
        test_num_frames.append(
            h5data["exps"][exp_names[i]]["labels"].shape[0]
        )



import pdb;pdb.set_trace()
