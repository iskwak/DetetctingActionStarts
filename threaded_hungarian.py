"""Mouse behavior spike classification."""
from __future__ import print_function, division
import os
import time
import sys
import gflags
import numpy

import h5py
import helpers.paths as paths
import helpers.arg_parsing as arg_parsing

import helpers.sequences_helper as sequences_helper
# import helpers.post_processing as post_processing
# import models.hantman_hungarian as hantman_hungarian
from models import hantman_hungarian
from helpers.hantman_sampler import HantmanVideoSampler
import flags.lstm_flags
import flags.cuda_flags
import torch

# python 2 vs 3 stuff...
if sys.version_info[0] < 3:
    import Queue as queue
else:
    import queue

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
gflags.MarkFlagAsRequired("feat_keys")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(hantman_hungarian)
gflags.ADOPT_module_key_flags(flags.lstm_flags)
gflags.ADOPT_module_key_flags(flags.cuda_flags)


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the number iterations per epoch.
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        num_train_vids = len(train_data["exp_names"])
        iter_per_epoch =\
            numpy.ceil(1.0 * num_train_vids / opts["flags"].hantman_mini_batch)

        iter_per_epoch = int(iter_per_epoch)
        opts["flags"].iter_per_epoch = iter_per_epoch
        opts["flags"].total_iterations =\
            iter_per_epoch * opts["flags"].total_epochs

    return opts


def _create_samplers(opts, train_data, test_data):
    train_sampler = HantmanVideoSampler(
        opts["rng"], train_data, opts["flags"].hantman_mini_batch,
        opts["flags"].seq_len, opts["flags"].feat_keys)
    test_sampler = HantmanVideoSampler(
        opts["rng"], test_data, opts["flags"].hantman_mini_batch,
        opts["flags"].seq_len, opts["flags"].feat_keys)

    return train_sampler, test_sampler


def _copy_templates(opts, train_data, test_data):
    print("copying frames/templates...")
    sequences_helper.copy_main_graphs(opts)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "train")
    # train_experiments = exp_list[train_vids]
    train_experiments = train_data["exp_names"].value
    sequences_helper.copy_experiment_graphs(
        opts, base_out, train_experiments)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "test")
    # test_experiments = exp_list[train_vids]
    test_experiments = test_data["exp_names"].value
    sequences_helper.copy_experiment_graphs(
        opts, base_out, test_experiments)


def _get_label_weight(opts, data):
    """Get number of positive examples for each label."""
    experiments = data["exp_names"].value
    label_mat = numpy.zeros((experiments.size, 7))
    vid_lengths = numpy.zeros((experiments.size,))
    for i in range(experiments.size):
        exp_key = experiments[i]
        exp = data["exps"][exp_key]
        for j in range(6):
            # label_counts[j] += exp["org_labels"].value[:, j].sum()
            label_mat[i, j] = exp["org_labels"].value[:, j].sum()
        # label_counts[-1] +=\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()
        label_mat[i, -1] =\
            exp["org_labels"].shape[0] - exp["org_labels"].value.sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = exp["org_labels"].shape[0]

    # label_counts = label_mat.sum(axis=0)
    label_weight = 1.0 / numpy.mean(label_mat, axis=0)
    # label_weight[-2] = label_weight[-2] * 10
    if opts["flags"].reweight is False:
        label_weight = [5, 5, 5, 5, 5, 5, .01]
    # import pdb; pdb.set_trace()
    return label_weight


def _init_network(opts, h5_data):
    """Setup the network."""
    exp_list = h5_data["exp_names"].value
    opts["feat_dims"] = [
        h5_data["exps"][exp_list[0]][feat_key].shape[2]
        for feat_key in opts["flags"].feat_keys
    ]
    # num_input = h5_data["exps"][exp_list[0]]["reduced"].shape[2]
    num_classes = h5_data["exps"][exp_list[0]]["labels"].shape[2]
    network = hantman_hungarian.HantmanHungarianConcat(
        input_dims=opts["feat_dims"],
        hidden_dim=opts["flags"].lstm_hidden_dim,
        output_dim=num_classes
    )

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
    """Train one epoch."""
    for sample in train_sampler:

        hidden = _get_hidden(opts, network)
        import pdb; pdb.set_trace()
        train_predict, update_hid = network(inputs, hidden)

        TP_weight, FP_weight, false_neg, false_pos = create_match_array(
            opts, train_predict, org_labels, label_weight[2])

        pos_mask, neg_mask = hantman_hungarian.create_pos_neg_masks(labels, label_weight[0], label_weight[1])
        perframe_cost = hantman_hungarian.perframe_loss(train_predict, mask, labels, pos_mask, neg_mask)
        tp_cost, fp_cost, fn_cost = hantman_hungarian.structured_loss(
            train_predict, mask, TP_weight, FP_weight, false_neg)

        total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost =\
            hantman_hungarian.combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost)
        cost = total_cost.mean()
        optimizer.zero_grad()
        cost.backward()
        # torch.nn.utils.clip_grad_norm(network.parameters(), 5)

        optimizer.step()
        step += 1

    return step


def _train_network(opts, network, optimizer,
                   train_sampler, test_sampler, label_weight):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        network.train()
        step = _train_epoch(opts, step, network, optimizer,
                            train_sampler, label_weight)
        print("\tFinished epoch")
        print("\tProcessing all examples...")
        network.eval()

        # round_tic = time.time()

    print("Finished training.")


def main(argv):
    print(argv)

    opts = _setup_opts(argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    full_tic = time.time()
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            # setup output space.
            tic = time.time()
            _copy_templates(opts, train_data, test_data)
            print(time.time() - tic)

            # get label weights
            tic = time.time()
            label_weight = _get_label_weight(opts, train_data)
            print(time.time() - tic)

            # create the data samplers.
            train_sampler, test_sampler = _create_samplers(
                opts, train_data, test_data)

            network, optimizer = _init_network(opts, train_data)

            _train_network(
                opts, network, optimizer,
                train_sampler, test_sampler, label_weight)
    print(time.time() - full_tic)


if __name__ == "__main__":
    main(sys.argv)
# # tic = time.time()
# num_batch = 0
# cpu_timing = 0
# gpu_timing = 0
# for sample in train_sampler:
#     # print(sample["exp_names"])
#     cpu_timing += sample["ram_time"]
#     gpu_timing += sample["gpu_time"]
#     num_batch += 1
# sample = train_sampler.get_rest()
# cpu_timing += sample["ram_time"]
# gpu_timing += sample["gpu_time"]
# print(cpu_timing / num_batch)
# print(gpu_timing / num_batch)

# num_batch = 0
# cpu_timing = 0
# gpu_timing = 0
# for sample in test_sampler:
#     # print(sample["exp_names"])
#     cpu_timing += sample["ram_time"]
#     gpu_timing += sample["gpu_time"]
#     num_batch += 1
# sample = test_sampler.get_rest()
# cpu_timing += sample["ram_time"]
# gpu_timing += sample["gpu_time"]
# print(cpu_timing / num_batch)
# print(gpu_timing / num_batch)
