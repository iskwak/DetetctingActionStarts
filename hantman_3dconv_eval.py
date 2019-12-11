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
# import helpers.post_processing as post_processing
import flags.cuda_flags
import torch
from models.hantman_3dconv import Hantman3DConv
from helpers.videosampler import HantmanVideoFrameSampler
from helpers.videosampler import HantmanVideoSampler
import helpers.git_helper as git_helper
# from torch.autograd import Variable

DEBUG = False
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
gflags.DEFINE_integer("update_iterations", None,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", None,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_string(
    "frames",
    "-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10",
    "Frames to process.")
# gflags.DEFINE_float(
#     "hantman_weight_decay", 0.0001, "Weight decay value.")
gflags.DEFINE_string("model", None, "Cached model.")
gflags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
gflags.DEFINE_integer(
    "hantman_mini_batch", 256,
    "Mini batch size for training.")

gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("train_file")
gflags.MarkFlagAsRequired("test_file")

# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)

g_label_names = [
    "lift", "hand", "grab", "supinate", "mouth", "chew"
]


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS
    opts = arg_parsing.setup_opts(argv, FLAGS)
    # this is dumb... passing in negative numbers to DEFINE_multi_int doesn't
    # seem to work well. So frames will be a string and split off of spaces.
    opts["flags"].frames = [
        int(frame_num) for frame_num in opts["flags"].frames.split(' ')
    ]

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


def _create_labels(opts, label_mat):
    label_idx = torch.LongTensor(label_mat.size(0))

    # pos_idx = label_mat.data.nonzero()
    # label_idx[:pos_idx.shape[0]] = pos_idx[:, 1]

    pos_idx = label_mat.nonzero()
    label_idx[:pos_idx.shape[0]] = pos_idx.data[:, 1]
    label_idx[pos_idx.shape[0]:] = 6

    if opts["flags"].cuda_device != -1:
        label_idx = label_idx.cuda()

    # label_idx = Variable(label_idx, requires_grad=False)

    return label_idx


def _init_network(opts, label_weight):
    """Setup the network."""
    # network = Hantman3DConv().cuda()
    network = Hantman3DConv()

    if opts["flags"].model is not None:
        model_dict = torch.load(opts["flags"].model)
        network.load_state_dict(model_dict)
    network = network.cuda()

    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    label_weight = torch.Tensor(label_weight)
    criterion = torch.nn.NLLLoss(label_weight)

    if opts["flags"].cuda_device != -1:
        network.cuda()
        criterion.cuda()

    return network, optimizer, criterion


def _proc_network(opts, network, optimizer, criterion,
                  sampler, train_eval, test_eval, valid_eval):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    frame_thresh = [
        10 for label in g_label_names
    ]
    step = 0
    print("\tProcessing all examples...")
    tic = time.time()
    network.eval()
    # train_cost = _eval_network(
    #     opts, step, network, train_eval, criterion, "train")

    test_cost = _eval_network(
        opts, step, network, test_eval, criterion, "test")
    # valid_cost = _eval_network(
    #     opts, step, network, valid_eval, criterion, "valid")

    # sequences_helper.log_outputs3(
    #     opts, step, train_cost, test_cost, valid_cost, g_label_names,
    #     frame_thresh=frame_thresh)

    print("\tProcessing finished: %f" % (time.time() - tic))
    print("Finished training.")


def _eval_network(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0

    num_behav = len(g_label_names)
    if sampler.batch_idx.empty():
        sampler.reset()
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()
        num_frames = inputs[0].size()[0]
        # chunk_len = 4

        # num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
        predict = np.zeros((num_frames, 1, num_behav))
        # print(num_frames)
        for j in range(0, num_frames):
            feat1, feat2, labels = _create_chunks(opts, inputs, j)
            chunk = [feat1, feat2, labels]

            if opts["flags"].cuda_device != -1:
                chunk = [
                    feat1.cuda(), feat2.cuda(), labels.cuda()
                ]

            out = network(chunk[:2])
            # if len((labels != 6).nonzero().size()):
            #     import pdb; pdb.set_trace()
            temp = torch.exp(out.data)

            predict[j, 0, :] = temp[:, :num_behav].cpu().numpy()

            loss = criterion(out, chunk[-1])
            total_loss += loss.item()
            # total_loss += loss.data[0]

        exp_names = [inputs[3][0]]
        labels = inputs[2].cpu().numpy()
        labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        frames = [range(labels[0].shape[0])]

        sequences_helper.write_predictions_list(out_dir, exp_names, predict,
                                                labels, None, frames,
                                                g_label_names)
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch)


def _load_chunk(opts, inputs, frames):
    feat1 = torch.zeros(1, 1, len(frames), 224, 224)
    feat2 = torch.zeros(1, 1, len(frames), 224, 224)

    if np.any(np.array(frames) < 0):
        for j in range(len(frames)):
            if frames[j] < 0:
                frames[j] = 0
    if np.any(np.array(frames) >= inputs[0].size(0)):
        for j in range(len(frames)):
            if frames[j] >= inputs[0].size(0):
                frames[j] = inputs[0].size(0) - 1
    idxs = range(0, 10)
    for i, frame in zip(idxs, frames):
        feat1[:, 0, i, :, :] = inputs[0][frame, 0, :, :]
        feat2[:, 0, i, :, :] = inputs[1][frame, 0, :, :]
    # import pdb; pdb.set_trace()
    return feat1, feat2


def _create_chunks(opts, inputs, frame_idx):
    """Create the overlapping chunks."""
    # idx2 = 75
    # idx1 = 71
    # num_batch = idx2 - idx1
    # img1 = torch.zeros(num_batch, 1, 10, 224, 224)
    # img2 = torch.zeros(num_batch, 1, 10, 224, 224)
    # labels = torch.zeros(num_batch)

    feat1_list = []
    feat2_list = []
    label_list = []

    frames = [
        offset + frame_idx for offset in opts["flags"].frames
    ]
    temp1, temp2 = _load_chunk(opts, inputs, frames)
    feat1_list.append(temp1)
    feat2_list.append(temp2)

    temp_label = inputs[2][frame_idx, :].nonzero()
    if len(temp_label.size()) == 1:
        temp_label = 6
    else:
        temp_label = temp_label[0][0]
    label_list.append(temp_label)

    feat1 = torch.cat(feat1_list, dim=0)
    feat2 = torch.cat(feat2_list, dim=0)
    labels = torch.LongTensor(label_list)
    return feat1, feat2, labels


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


def main(argv):
    opts = _setup_opts(sys.argv)
    # paths.setup_output_space(opts)
    out_dir = os.path.join(opts["flags"].out_dir, 'proc_info')
    paths.create_dir(out_dir)
    paths.save_command(opts, out_dir)
    git_helper.log_git_status(os.path.join(out_dir, "git_status.txt"))

    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            with h5py.File(opts["flags"].valid_file, "r") as valid_data:

                sampler = HantmanVideoFrameSampler(
                    opts["rng"], train_data, opts["flags"].video_dir,
                    opts["flags"].hantman_mini_batch,
                    frames=opts["flags"].frames,
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

                network, optimizer, criterion = _init_network(
                    opts, label_weight)
                # import pdb; pdb.set_trace()
                _proc_network(
                    opts, network, optimizer, criterion,
                    sampler, train_eval, test_eval, valid_eval
                )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
