"""Create 3d conv features."""
import sys
import os
import h5py
import numpy
# from helpers.RunningStats import RunningStats
import helpers.paths as paths
import helpers.git_helper as git_helper
from models.hantman_3dconv import Hantman3DConv
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from collections import OrderedDict
import cv2
import PIL
import time
# import torch.nn.Parameter as Parameter

input_name = "/nrs/branson/kwaki/data/20170718/one_mouse_multi_day_test.hdf5"
out_dir = "/media/drive2/kwaki/data/hantman_processed/20170827_vgg_transpose"


def reshape_feat(features):
    temp = features.reshape(features.shape[0], 1, features.shape[1])
    return temp


def process_exp(in_data, exp_name):
    # process the experiment, just transposing the values
    exp_filename = os.path.join(out_dir, "exps", exp_name)
    with h5py.File(exp_filename, "w") as exp_file:
        exp_file["date"] = in_data["exps"][exp_name]["data"].value
        exp_file["labels"] = in_data["exps"][exp_name]["labels"].value
        exp_file["org_labels"] = in_data["exps"][exp_name]["org_labels"].value
        
        temp = reshape_feat(in_data["exps"][exp_name]["hoghof"].value)
        exp_file["hoghof"] = temp

        temp = reshape_feat(in_data["exps"][exp_name]["hoghof_norm"].value)
        exp_file["hoghof_norm"] = temp

        temp = reshape_feat(in_data["exps"][exp_name]["pos_features"].value)
        exp_file["pos_features"] = temp

        temp = reshape_feat(in_data["exps"][exp_name]["pos_norm"].value)
        exp_file["pos_norm"] = temp

        # reduced is already the right dimensions... so confusing.
        exp_file["reduced"] = in_data["exps"][exp_name]["reduced"].value


def main():
    output_name = os.path.join(out_dir, "one_mouse_multi_day_test.hdf5")
    with h5py.File(input_name, "r") as in_data:
        exp_names = in_data["exp_names"].value
        exp_names.sort()

        with h5py.File(output_name, "w") as out_data:
            # setup the keys to match the input hdf5
            out_data["date"] = in_data["date"].value
            out_data["exp_names"] = in_data["exp_names"].value
            out_data["experiments"] = in_data["experiments"].value
            out_data["mice"] = in_data["mice"].value
            out_data.create_group("exps")

            for exp in exp_names:
                print(exp)
                process_exp(in_data, exp)
                # import pdb; pdb.set_trace()
                out_data["exps"][exp] = h5py.ExternalLink(
                    os.path.join("exps", exp), "/"
                )


if __name__ == "__main__":
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)
    git_helper.log_git_status(
        os.path.join(out_dir, "00_git_status.txt"))

    main()
