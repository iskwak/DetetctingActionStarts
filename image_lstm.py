"""Mouse behavior spike classification."""
from __future__ import print_function, division
import os
import time
import sys
import gflags
# import numpy as np

import h5py
import helpers.paths as paths
import helpers.arg_parsing as arg_parsing

import helpers.general as general
# import models.hantman_hungarian as hantman_hungarian
# from models import hantman_hungarian
import flags.lstm_flags
import flags.cuda_flags
import torch
from helpers.hantman_sampler import HantmanFrameSeqSampler
from helpers.hantman_sampler import HantmanVideoFrameSampler
# import torchvision.models as models
# import models.hantman_feedforward as hantman_feedforward
import models.lstm_conv2d as lstm_conv2d
# from torch.autograd import Variable
# from models import hantman_hungarian

# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("image_dir", None, "Directory for images to symlink.")
gflags.DEFINE_integer("total_iterations", 0,
                      "Don't set for this version of the training code.")
# gflags.DEFINE_boolean("debug", False, "Debug flag, work with less videos.")
gflags.DEFINE_integer("update_iterations", None,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", None,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_boolean("threaded", True, "Threaded Data loadered.")
gflags.DEFINE_boolean("anneal", True, "Use annealing on perframe cost.")
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_list("feat_keys", None, "Feature keys to use.")
gflags.DEFINE_string("arch", "concat", "Which lstm arch to use.")
# gflags.DEFINE_float(
#     "hantman_weight_decay", 0.0001, "Weight decay value.")

gflags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
gflags.DEFINE_integer(
    "hantman_mini_batch", 256,
    "Mini batch size for training.")
gflags.DEFINE_integer("hantman_seq_length", 1500, "Sequence length.")


gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("train_file")
gflags.MarkFlagAsRequired("test_file")

# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)
# gflags.ADOPT_module_key_flags(hantman_hungarian)
# gflags.ADOPT_module_key_flags(flags.lstm_flags)


def _init_model(opts, label_weight):
    network = lstm_conv2d.HantmanHungarianImage()
    # network = hantman_feedforward.HantmanFeedForwardVGG(pretrained=True)
    if opts["flags"].cuda_device != -1:
        # put on the GPU for better compute speed
        network.cuda()

    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    if opts["flags"].cuda_device != -1:
        network.cuda()

    return network, optimizer


def _get_hidden(opts, network):
    if opts["flags"].cuda_device >= 0:
        use_cuda = True
    else:
        use_cuda = False

    hidden = network.init_hidden(
        opts["flags"].hantman_mini_batch,
        use_cuda=use_cuda)
    return hidden


def _train_epoch(opts, step, network, optimizer, train_sampler, label_weight):
    network.train()

    # for i in range(100):
    total_loss = 0
    for i in range(train_sampler.num_batch):
        # print("sampling")
        blob = train_sampler.get_minibatch()
        hidden = _get_hidden(opts, network)
        # img_side = torch.autograd.Variable(torch.Tensor(inputs[0])).cuda()
        # img_front = torch.autograd.Variable(torch.Tensor(inputs[1])).cuda()
        #import pdb; pdb.set_trace()
        out = network(blob["features"][0], hidden)
        import pdb; pdb.set_trace()
        loss = criterion(out, inputs[-1])
        total_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # import pdb; pdb.set_trace()
    return total_loss


def _train_network(opts, network, optimizer, train_sampler,
                   train_eval, test_eval, label_weight):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        network.train()
        total_loss = _train_epoch(opts, step, network, optimizer, train_sampler, label_weight)
        step += train_sampler.num_batch
        print("\tFinished epoch")
        print("\tProcessing all examples...")
        print("\tave loss: %f" % (total_loss / train_sampler.num_batch))
        network.eval()
        train_cost = general.eval_network(
            opts, step, network, label_weight, train_eval, "train")
        test_cost = general.eval_network(
            opts, step, network, label_weight, test_eval, "test")
        general.log_outputs(opts, step, train_cost, test_cost)
        # round_tic = time.time()

        # save the network in its own folder in the networks folder
        out_dir = os.path.join(
            opts["flags"].out_dir, "networks", "%d" % step)
        paths.create_dir(out_dir)
        out_name = os.path.join(out_dir, "network.pt")
        tic = time.time()
        torch.save(network.cpu().state_dict(), out_name)
        network.cuda()
        print("\t%f" % (time.time() - tic))

    print("Finished training.")


def main(argv):
    opts = general.setup_opts(sys.argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    # load data
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            general.copy_templates(opts, train_data, test_data)
            train_sampler = HantmanFrameSeqSampler(
                opts["rng"], train_data, "/media/drive1/data/hantman_frames",
                opts["flags"].seq_len, opts["flags"].hantman_mini_batch,
                use_pool=True, max_queue=1, gpu_id=opts["flags"].cuda_device)
            train_eval = HantmanVideoFrameSampler(
                train_data, "/media/drive1/data/hantman_frames",
                use_pool=False, gpu_id=opts["flags"].cuda_device)
            test_eval = HantmanVideoFrameSampler(
                test_data, "/media/drive1/data/hantman_frames",
                use_pool=False, gpu_id=opts["flags"].cuda_device)

            label_weight = general.get_label_weight(opts, train_data)
            model, optimizer = _init_model(opts, label_weight)
            _train_network(opts, model, optimizer,
                           train_sampler, train_eval, test_eval,
                           label_weight)
            print("stuff")
    print("moo")


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
