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
from torch.autograd import Variable
from models.jigsaw_3dconv import Jigsaw3DConv
from helpers.videosampler import VideoFrameSampler
from helpers.videosampler import VideoSampler
from helpers.RunningStats import RunningStats
import torchvision.transforms as transforms
import models.jigsaw_2dconv as jigsaw_2dconv

# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("image_dir", None, "Directory for images to symlink.")
gflags.DEFINE_string("video_dir", None, "Directory for processing (org) videos.")
gflags.DEFINE_string("display_dir", None, "Directory for display videos (mp4).")
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
# gflags.DEFINE_float(
#     "hantman_weight_decay", 0.0001, "Weight decay value.")
gflags.DEFINE_boolean("normalize", False, "Normalize data.")
gflags.DEFINE_list("frames", [0], "List of frame offsets")

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
    ("G%02d" % i) for i in range(1, 16)
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

    # convert the frames list into a list of values
    opts["flags"].frames = [
        int(frame) for frame in opts["flags"].frames
    ]

    return opts


def _convert_labels(labels):
    # relabel = torch.zeros(labels.size()[0], dtype=torch.int64)
    relabel = torch.LongTensor(labels.size()[0]).zero_()
    labels = labels.cpu().numpy()
    for i in range(labels.shape[0]):
        idx = np.argwhere(labels[i, :] == 1)
        if idx.size > 0:
            idx = idx[0][0]
        else:
            idx = 15
        relabel[i] = idx
    # relabel = torch.Tensor(relabel)
    return relabel


def _train_epoch(opts, network, optimizer, criterion, sampler):

    print(sampler.num_batch)
    if sampler.batch_idx.empty():
        sampler.reset()
    # for i in range(100):
    for i in range(sampler.num_batch):
        print("\t%d" % i)
        tic = time.time()
        batch = sampler.get_minibatch()
        labels = _convert_labels(batch[1])
        features = batch[0].cuda()
        # cow[1] = cow[1].cuda()
        # convert to variables
        inputs = Variable(features, requires_grad=True).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        # print("\tdata: %f" % (time.time() - tic))

        tic = time.time()
        out = network(inputs)
        loss = criterion(out, labels)
        # import pdb; pdb.set_trace()
        # print("\tforward: %f" % (time.time() - tic))

        tic = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        boo, cow = torch.exp(out.data).max(1)
        # print(cow)
        # print(labels)
        # print(cow != labels.data)
        # print("\tbackward: %f" % (time.time() - tic))

    # import pdb; pdb.set_trace()
    # print("moo")
    return sampler.num_batch


def _train_network(opts, network, optimizer, criterion, train_data, test_data,
                   sampler, train_eval, test_eval):
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
        print("\tProcessing all examples...")

        # import pdb; pdb.set_trace()
        if i % 5 == 0:
            tic = time.time()
            network.eval()
            train_cost = _eval_network(
                opts, step, network, train_eval, criterion, "train")
            test_cost = _eval_network(
                opts, step, network, test_eval, criterion, "test")
            sequences_helper.log_outputs2(
                opts, step, train_cost, test_cost, g_label_names,
                frame_thresh=frame_thresh)
            # print("\teval time: %f" % (time.time() - tic))
        # network.eval()
        # _log_outputs(opts, step, network, label_weight)
        # round_tic = time.time()

        # # save the network in its own folder in the networks folder
        # out_dir = os.path.join(
        #     opts["flags"].out_dir, "networks", "%d" % step)
        # paths.create_dir(out_dir)
        # out_name = os.path.join(out_dir, "network.pt")
        # torch.save(network.cpu().state_dict(), out_name)
        # network.cuda()
        # # hantman_hungarian_image.save_network(opts, network, out_name)
    print("Finished training.")


def _eval_network(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0

    if sampler.batch_idx.empty():
        sampler.reset()
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()
        num_frames = inputs[0].size()[0]
        chunk_len = 1
        num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
        predict = np.zeros((num_frames, 1, 16))
        # print(num_frames)
        for j in range(0, num_chunks):
            idx1 = j * chunk_len
            idx2 = min((j + 1) * chunk_len, num_frames)
            # print("\t%d" % idx2)
            feat1, labels = _create_chunks(opts, inputs, idx1, idx2)
            chunk = [
                Variable(feat1, requires_grad=True).cuda(),
                Variable(labels, requires_grad=False).cuda()
            ]

            out = network(chunk[0])
            # if len((labels != 6).nonzero().size()):
            #     import pdb; pdb.set_trace()
            temp = torch.exp(out.data)
            import pdb; pdb.set_trace()
            predict[idx1:idx2, 0, :] = temp[:, :16].cpu().numpy()

            loss = criterion(out, chunk[-1])
            total_loss += loss.data[0]

        exp_names = [inputs[2][0]]
        labels = inputs[1].cpu().numpy()
        labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        frames = [range(labels[0].shape[0])]
        # import pdb; pdb.set_trace()
        sequences_helper.write_predictions_list(out_dir, exp_names, predict,
                                                labels, None, frames,
                                                g_label_names)
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch)


def _load_chunk(opts, inputs, frames):
    feat1 = torch.zeros(10, 3, 224, 224)

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
        feat1[i, :, :, :] = inputs[0][frame, :, :, :]
    # import pdb; pdb.set_trace()
    return feat1


def _create_chunks(opts, inputs, idx1, idx2):
    """Create the overlapping chunks."""
    # idx2 = 75
    # idx1 = 71
    num_batch = idx2 - idx1
    # img1 = torch.zeros(num_batch, 1, 10, 224, 224)
    # img2 = torch.zeros(num_batch, 1, 10, 224, 224)
    # labels = torch.zeros(num_batch)

    feat1_list = []
    label_list = []
    for i in range(num_batch):
        curr_idx = i + idx1
        frames = range(curr_idx - 5, curr_idx + 5)
        temp1 = _load_chunk(opts, inputs, frames)
        feat1_list.append(temp1)

        temp_label = inputs[1][curr_idx, :].nonzero()
        if len(temp_label.size()) == 0:
            temp_label = 6
        else:
            if temp_label.size()[0] != 0:
                temp_label = temp_label[0][0]
                label_list.append(temp_label)

    feat1 = torch.cat(feat1_list, dim=0)
    labels = torch.LongTensor(label_list)
    return feat1, labels


def _copy_templates(opts, train_data, test_data):
    print("copying frames/templates...")
    sequences_helper.copy_main_graphs(opts)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "train")
    # train_experiments = exp_list[train_vids]
    # train_experiments = train_data["exp_names"].value
    sequences_helper.copy_movie_graphs(
        opts, base_out, train_data, g_label_names
    )

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "test")
    # test_experiments = exp_list[train_vids]
    # test_experiments = test_data["exp_names"].value
    # sequences_helper.copy_experiment_graphs(
    #     opts, base_out, test_experiments)
    sequences_helper.copy_movie_graphs(
        opts, base_out, test_data, g_label_names
    )
    # import pdb; pdb.set_trace()


def _init_network(opts):
    """Setup the network."""
    if len(opts["flags"].frames) == 1:
        # if only 1 frame provided, 2d conv net.
        network = jigsaw_2dconv.jigsaw_2dconv(16)
    else:
        # else its a list, so 3d conv net.
        network = Jigsaw3DConv()

    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    criterion = torch.nn.NLLLoss()

    if opts["flags"].cuda_device != -1:
        network.cuda()
        criterion.cuda()

    return network, optimizer, criterion


def compute_means(opts, train_data, sampler):
    """Go over the features and compute the mean and variance."""
    exp_names = train_data["exp_names"].value
    means = []
    stds = []
    if opts["flags"].normalize is True:
        running_stats = []
        # a running stat for each channel
        running_stats = RunningStats(3)
        # loop over the experiments

        # for exp_name in exp_names:
        for j in range(0, len(exp_names), 2):
            batch = sampler.get_minibatch()
            exp_name = batch[2][0]
            print(exp_name)
            # loop over the keys

            seq_len = train_data["exps"][exp_name]["labels"].shape[0]
            temp_feat = batch[0].cpu().numpy()
            temp_feat = temp_feat[:seq_len, :, :, :]

            channel_feats = []
            for i in range(3):
                # channel_feat = temp_feat[0, :, i, :]
                # sample frames
                channel_feat = temp_feat[::100, i, :]
                channel_feat = channel_feat.reshape(-1, 1)
                channel_feats.append(channel_feat)

            channel_feats = np.concatenate(channel_feats, axis=1)
            running_stats.add_data(
                channel_feat
            )

        means = running_stats.mean.tolist()
        stds = running_stats.compute_std().tolist()
    else:
        means = [.5, .5, .5]
        stds = [1, 1, 1]
        # for key in opts["flags"].feat_keys:
        #     temp_feat = train_data["exps"][exp_names[0]][key].value
        #     mean = np.zeros((temp_feat.shape[2], ))
        #     std = np.ones((temp_feat.shape[2], ))
        #     means.append(mean)
        #     stds.append(std)
    normalize = transforms.Normalize(mean=means,
                                     std=stds)

    return normalize


def main(argv):
    opts = _setup_opts(sys.argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            _copy_templates(opts, train_data, test_data)
            print("done")

            print("computing means")
            tic = time.time()
            temp_sampler = VideoSampler(
                opts["rng"], train_data, opts["flags"].video_dir, seq_len=2000,
                use_pool=False, gpu_id=opts["flags"].cuda_device)
            temp_sampler.reset()
            normalize = compute_means(opts, train_data, temp_sampler)
            print(time.time() - tic)

            sampler = VideoFrameSampler(
                opts["rng"], train_data, opts["flags"].video_dir,
                opts["flags"].hantman_mini_batch,
                frames=opts["flags"].frames, use_pool=False,
                gpu_id=opts["flags"].cuda_device, normalize=normalize)
                # frames=range(-10, 11), use_pool=False,
                # gpu_id=opts["flags"].cuda_device, normalize=normalize)

            train_eval = VideoSampler(
                opts["rng"], train_data, opts["flags"].video_dir, seq_len=-1,
                use_pool=False, gpu_id=opts["flags"].cuda_device, normalize=normalize)
            test_eval = VideoSampler(
                opts["rng"], test_data, opts["flags"].video_dir, seq_len=-1,
                use_pool=False, gpu_id=opts["flags"].cuda_device, normalize=normalize)

            network, optimizer, criterion = _init_network(opts)

            # batch = sampler.get_minibatch()
            # import pdb; pdb.set_trace()
            tic = time.time()
            # network(Variable(batch[0], requires_grad=True))

            _train_network(opts, network, optimizer, criterion,
                           train_data, test_data,
                           sampler, train_eval, test_eval)

            print(time.time() - tic)
            import pdb; pdb.set_trace()
            print("moo")


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
