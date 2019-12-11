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

DEBUG = True
# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("valid_file", None, "Valid data filename (hdf5).")
gflags.DEFINE_string("display_dir", None, "Directory of videos for display.")
gflags.DEFINE_string(
    "video_dir", None,
    "Directory for processing videos, (codecs might be different from display)")
gflags.DEFINE_integer("total_iterations", 0,
                      "Don't set for this version of the training code.")
# gflags.DEFINE_boolean("debug", False, "Debug flag, work with less videos.")
gflags.DEFINE_integer("update_iterations", 50,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", 10,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_boolean("threaded", True, "Threaded Data loadered.")
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
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

g_label_names = [
    "lift", "hand", "grab", "suppinate", "mouth", "chew"
]


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the number iterations per epoch.
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        num_train_vids = len(train_data["exp_names"])
        iter_per_epoch =\
            np.ceil(1.0 * num_train_vids / opts["flags"].hantman_mini_batch)

        iter_per_epoch = int(iter_per_epoch)
        opts["flags"].iter_per_epoch = iter_per_epoch
        opts["flags"].total_iterations =\
            iter_per_epoch * opts["flags"].total_epochs

    return opts


def _get_label_weight(opts, data):
    """Get number of positive examples for each label."""
    tic = time.time()
    experiments = data["exp_names"].value
    label_mat = np.zeros((experiments.size, 7))
    vid_lengths = np.zeros((experiments.size,))
    for i in range(experiments.size):
        exp_key = experiments[i]
        exp = data["exps"][exp_key]
        for j in range(6):
            # label_counts[j] += exp["org_labels"].value[:, j].sum()
            label_mat[i, j] = exp["labels"].value[:, j].sum()
        # label_counts[-1] +=\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()
        label_mat[i, -1] =\
            exp["labels"].shape[0] - exp["labels"].value.sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = exp["labels"].shape[0]

    # label_counts = label_mat.sum(axis=0)
    # import pdb; pdb.set_trace()

    label_weight = 1.0 / np.mean(label_mat, axis=0)
    # label_weight = label_mat.sum(axis=0)  / np.mean(label_mat, axis=0)
    # label_weight[-2] = label_weight[-2] * 10
    if opts["flags"].reweight is False:
        label_weight = [5, 5, 5, 5, 5, 5, .01]
    print(time.time() - tic)
    return label_weight


def _init_model(opts, label_weight):
    network = hantman_feedforward.HantmanFeedForward(pretrained=False)
    # network = hantman_feedforward.HantmanFeedForwardVGG(pretrained=True)
    if opts["flags"].cuda_device != -1:
        # put on the GPU for better compute speed
        network.cuda()

    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()

    label_weight = torch.Tensor(label_weight)
    criterion = torch.nn.NLLLoss(label_weight)

    if opts["flags"].cuda_device != -1:
        network.cuda()
        criterion.cuda()
    # import pdb; pdb.set_trace()
    return network, optimizer, criterion


def _create_labels(opts, label_mat):
    label_idx = torch.LongTensor(label_mat.size(0))

    # pos_idx = label_mat.data.nonzero()
    # label_idx[:pos_idx.shape[0]] = pos_idx[:, 1]

    pos_idx = label_mat.nonzero()
    if pos_idx.shape[0] > 0:
        label_idx[:pos_idx.shape[0]] = pos_idx.data[:, 1]
    label_idx[pos_idx.shape[0]:] = 6

    if opts["flags"].cuda_device != -1:
        label_idx = label_idx.cuda()

    # label_idx = Variable(label_idx, requires_grad=False)

    return label_idx


def _train_epoch(opts, network, optimizer, criterion, sampler):
    sampler.reset()
    sampling = []
    gpuing = []

    for i in range(sampler.num_batch):
        # print("%d of %d" % (i, sampler.num_batch))
        tic = time.time()
        blob = sampler.get_minibatch()

        label_idx = _create_labels(opts, blob[2])
        sampling.append(time.time() - tic)

        tic = time.time()
        out, features = network(blob[:2])

        loss = criterion(out, label_idx)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        gpuing.append(time.time() - tic)
        # torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()

    print("\t%f" % np.mean(sampling))
    print("\t%f" % np.mean(gpuing))
    # print("\t%f" % np.min(sampling))

    return sampler.num_batch


def _eval_network(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0
    if sampler.batch_idx.empty():
        sampler.reset()
    # sampler actually only provides a full movie.
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()

        total_loss += _eval_video(opts, network, inputs, criterion, name, out_dir)

        # evaluate the tp/fp's.
        # import pdb; pdb.set_trace()
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch)


def _eval_video(opts, network, inputs, criterion, name, out_dir):
    """Evaluate one video."""
    num_frames = inputs[0].size()[0]
    chunk_len = 50
    num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
    predict = np.zeros((num_frames, 1, 6))

    total_loss = 0
    # loop over the number of chunks
    for j in range(0, num_chunks):
        idx1 = j * chunk_len
        idx2 = min((j + 1) * chunk_len, num_frames)
        # print("\t%d" % idx2)

        chunk = [
            inputs[0][idx1:idx2, :1, :, :].cuda(),
            inputs[1][idx1:idx2, :1, :, :].cuda(),
            inputs[2][idx1:idx2, :].cuda()
        ]
        out, _ = network([chunk[0], chunk[1]])
        out = torch.exp(out.data)

        predict[idx1:idx2, 0, :] = out[:, :6].data.cpu().numpy()
        label_idx = _create_labels(opts, chunk[-1])

        loss = criterion(out, label_idx)
        total_loss += loss.item()

    # finished predicting the video.
    exp_names = [inputs[3]]
    labels = inputs[2].cpu().numpy()
    labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
    frames = [range(labels[0].shape[0])]

    # write the predictions to disk
    sequences_helper.write_predictions2(
        out_dir, exp_names[0], predict, labels, None, frames)

    # process the video
    return total_loss


def _train_network(opts, network, optimizer, criterion,
                   sampler, train_eval, test_eval, valid_eval):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    frame_thresh = [
        10 for label in g_label_names
    ]
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        tic = time.time()
        network.train()
        step += _train_epoch(opts, network, optimizer, criterion, sampler)
        print("\t%f" % (time.time() - tic))
        print("\tFinished epoch")

        if i % opts['flags'].update_iterations == 0 and i != 0:
            print("\tProcessing all examples...")
            tic = time.time()
            network.eval()
            train_cost = _eval_network(
                opts, step, network, train_eval, criterion, "train")
            if DEBUG:
                test_cost = train_cost
                valid_cost = train_cost
            else:
                test_cost = _eval_network(
                    opts, step, network, test_eval, criterion, "test")
                valid_cost = _eval_network(
                    opts, step, network, valid_eval, criterion, "valid")

            sequences_helper.log_outputs3(
                opts, step, train_cost, test_cost, valid_cost, g_label_names,
                frame_thresh=frame_thresh)
        if i % opts['flags'].save_iterations == 0:
            # save the network in its own folder in the networks folder
            print("\tSaving network...")
            out_dir = os.path.join(
                opts["flags"].out_dir, "networks", "%d" % step)
            paths.create_dir(out_dir)
            out_name = os.path.join(out_dir, "network.pt")
            torch.save(network.cpu().state_dict(), out_name)
            network.cuda()
        print("\tProcessing finished: %f" % (time.time() - tic))

    out_dir = os.path.join(
        opts["flags"].out_dir, "networks", "%d" % step)
    paths.create_dir(out_dir)
    out_name = os.path.join(out_dir, "network.pt")
    torch.save(network.cpu().state_dict(), out_name)
    network.cuda()
    print("Finished training.")


def main(argv):
    opts = _setup_opts(argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    # load data
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            with h5py.File(opts["flags"].valid_file, "r") as valid_data:
                if DEBUG:
                    train_data = valid_data
                    test_data = valid_data

                sequences_helper.copy_templates(
                    opts, train_data, "train", g_label_names)
                sequences_helper.copy_templates(
                    opts, test_data, "test", g_label_names)
                sequences_helper.copy_templates(
                    opts, valid_data, "valid", g_label_names
                )

                sampler = HantmanVideoFrameSampler(
                    opts["rng"], train_data, opts["flags"].video_dir,
                    opts["flags"].hantman_mini_batch,
                    frames=[0],
                    use_pool=True, gpu_id=opts["flags"].cuda_device)

                label_weight = _get_label_weight(opts, train_data)
                # import pdb; pdb.set_trace()
                train_eval = HantmanVideoSampler(
                    None, train_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)
                test_eval = HantmanVideoSampler(
                    None, test_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)
                valid_eval = HantmanVideoSampler(
                    None, valid_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)

                # import pdb; pdb.set_trace()
                network, optimizer, criterion = _init_model(opts, label_weight)

                _train_network(
                    opts, network, optimizer, criterion,
                    sampler, train_eval, test_eval, valid_eval
                )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
