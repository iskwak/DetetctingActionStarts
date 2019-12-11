"""Copy hdf5 data."""
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

input_name = "/media/drive2/kwaki/data/hantman_processed/20180206_imgs/one_mouse_multi_day_train.hdf5"
out_dir = "/media/drive3/kwaki/data/hantman_processed/20180305_imgs"


def process_exp(exp_name, in_file, out_dir):
    exp = in_file["exps"][exp_name]

    img_side = exp["raw"]["img_side"].value
    img_front = exp["raw"]["img_front"].value
    labels = exp["raw"]["labels"].value

    # hack... need to get processed labels. question, is this still needed?
    label_filename = os.path.join(
        "/media/drive1/data/hantman_processed/20170827_vgg/exps",
        exp_name)
    with h5py.File(label_filename, "r") as label_data:
        proc_labels = label_data["labels"].value

    # save to the new experiment file.
    out_name = os.path.join(out_dir, "exps", exp_name)
    with h5py.File(out_name, "w") as out_file:
        out_file["img_side"] = img_side
        out_file["img_front"] = img_front
        out_file["date"] = exp["date"].value
        out_file["mouse"] = exp["mouse"].value
        out_file["org_labels"] = labels
        out_file["labels"] = proc_labels


def main():
    output_name = os.path.join(out_dir, "one_mouse_multi_day_train.hdf5")

    with h5py.File(input_name, "r") as in_file:
        with h5py.File(output_name, "w") as out_file:
            # copy over the general stats
            print("a")
            out_file["date"] = in_file["date"].value
            out_file["exp_names"] = in_file["exp_names"].value
            out_file["mice"] = in_file["mice"].value
            print("b")
            out_file.create_group("exps")
            for exp_name in out_file["exp_names"]:
                print(exp_name)
                tic = time.time()
                # process_exp(exp_name, in_file, out_dir)
                out_file["exps"][exp_name] = h5py.ExternalLink(
                    os.path.join("exps", exp_name), "/"
                )
                print("\t%f" % (time.time() - tic))


if __name__ == "__main__":
    paths.create_dir(out_dir)
    paths.create_dir(os.path.join(out_dir, "exps"))
    paths.save_command2(out_dir, sys.argv)
    git_helper.log_git_status(
        os.path.join(out_dir, "00_git_status.txt"))

    main()
