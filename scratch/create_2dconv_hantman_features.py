"""Mouse behavior spike classification."""
from __future__ import print_function, division
import os
import time
import sys
import gflags
import numpy as np

import h5py
import helpers.paths as paths
import helpers.arg_parsing as arg_parsing

import helpers.sequences_helper as sequences_helper
import helpers.post_processing as post_processing
# import models.hantman_hungarian as hantman_hungarian
# from models import hantman_hungarian
import flags.lstm_flags
import flags.cuda_flags
import torch
from helpers.videosampler import HantmanVideoFrameSampler
from helpers.videosampler import HantmanVideoSampler
# import torchvision.models as models
import models.hantman_feedforward as hantman_feedforward
from torch.autograd import Variable
import helpers.git_helper as git_helper


DEBUG = False
# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("input_name", None, "Input file to create features for.")
gflags.DEFINE_string(
    "video_dir", None,
    "Directory for processing videos, (codecs might be different from display)")
gflags.DEFINE_string("load_network", None, "Cached network to load.")

gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("input_name")
gflags.MarkFlagAsRequired("video_dir")

# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)

g_label_names = [
    "lift", "hand", "grab", "suppinate", "mouth", "chew"
]


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    return opts


def _init_model(opts):
    state_dict = torch.load(opts["flags"].load_network)

    network = hantman_feedforward.HantmanFeedForward(pretrained=False)
    network.load_state_dict(state_dict)

    # network = hantman_feedforward.HantmanFeedForwardVGG(pretrained=True)
    if opts["flags"].cuda_device != -1:
        # put on the GPU for better compute speed
        network.cuda()

    if opts["flags"].cuda_device != -1:
        network.cuda()
    # import pdb; pdb.set_trace()
    return network


def process_exps(opts, sampler, network, out_data, in_data):
    """Evaluate the state of the network."""
    if sampler.batch_idx.empty():
        sampler.reset()
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()
        print(inputs[-1][0])

        num_frames = inputs[0].size()[0]
        chunk_len = 50
        num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))

        all_features = torch.zeros(num_frames, 1024)
        for j in range(0, num_chunks):
            idx1 = j * chunk_len
            idx2 = min((j + 1) * chunk_len, num_frames)
            # print("\t%d" % idx2)
            chunk = [
                inputs[0][idx1:idx2, :1, :, :].cuda(),
                inputs[1][idx1:idx2, :1, :, :].cuda(),
                inputs[2][idx1:idx2, :].cuda()
            ]

            out, features = network([chunk[0], chunk[1]])
            features = features.cpu()
            all_features[idx1:idx2, :] = features.data
            # all_features[idx1:idx2, :] = features

        # tic = time.time()
        exp = in_data["exps"][inputs[-1][0]]
        out_name = os.path.join(opts["flags"].out_dir, "exps", inputs[-1][0])
        with h5py.File(out_name, "w") as exp_data:
            exp_data["conv2d"] = all_features.numpy()
            exp_data["date"] = exp["date"].value
            exp_data["mouse"] = exp["mouse"].value
            exp_data["labels"] = exp["labels"].value
            exp_data["video_name"] = exp["video_name"].value

        out_data["exps"][inputs[-1][0]] = h5py.ExternalLink(
            os.path.join("exps", inputs[-1][0]), "/")
    # import pdb; pdb.set_trace()
    return


def main(argv):
    opts = _setup_opts(argv)

    paths.create_dir(opts["flags"].out_dir)
    paths.create_dir(os.path.join(opts["flags"].out_dir, "exps"))
    paths.save_command2(opts["flags"].out_dir, sys.argv)
    git_helper.log_git_status(
        os.path.join(opts["flags"].out_dir, "00_git_status.txt"))

    output_name = os.path.join(
        opts["flags"].out_dir,
        os.path.basename(opts["flags"].input_name))

    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    # load data
    with h5py.File(opts["flags"].input_name, "r") as in_data:
        with h5py.File(output_name, "w") as out_data:
            sampler = HantmanVideoSampler(
                None, in_data, opts["flags"].video_dir,
                use_pool=True, gpu_id=opts["flags"].cuda_device)

            # import pdb; pdb.set_trace()
            network = _init_model(opts)
            network.eval()

            out_data["date"] = in_data["date"].value
            out_data["exp_names"] = in_data["exp_names"].value
            out_data["mice"] = in_data["mice"].value
            out_data.create_group("exps")

            process_exps(opts, sampler, network, out_data, in_data)

if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
