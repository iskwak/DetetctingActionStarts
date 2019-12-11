"""Apply hungarian loss on lstm."""
from __future__ import print_function, division
import sys
import time
import h5py
import gflags
import numpy
import torch
from scipy import signal

import helpers.arg_parsing as arg_parsing
import flags.lstm_flags
import flags.cuda_flags
import helpers.paths as paths
import helpers.sequences_helper as sequences_helper
from helpers.videosampler import HDF5Sampler
from models import hantman_hungarian
from helpers.RunningStats import RunningStats

# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("valid_file", None, "Valid data filename (hdf5).")
gflags.DEFINE_string("display_dir", None, "Directory of videos for display.")
# gflags.DEFINE_string(
#     "video_dir", None,
#     "Directory for processing videos, (codecs might be different from display)")
gflags.DEFINE_integer("total_iterations", 0,
                      "Don't set for this version of the training code.")
gflags.DEFINE_integer("update_iterations", 50,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", 10,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_integer("seq_len", 4000, "Sequence length.")
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_boolean("threaded", True, "Threaded Data loadered.")
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_boolean("anneal", True, "Use annealing on perframe cost.")
gflags.DEFINE_list("feat_keys", None, "Feature keys to use.")
gflags.DEFINE_string("arch", "concat", "Which lstm arch to use.")
gflags.DEFINE_string("split", None, "Which split to process. Can be empty.")

gflags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
gflags.DEFINE_integer("mini_batch", 256, "Mini batch size for training.")
gflags.DEFINE_boolean("normalize", False, "Normalize data.")

gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("train_file")
gflags.MarkFlagAsRequired("test_file")
# validation file isn't required

# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)

g_label_names = [
    ("G%02d" % i) for i in range(1, 16)
]


def _get_label_weight(train_sampler):
    """Get number of positive examples for each label."""
    # experiments = data["exp_names"].value
    experiments = train_sampler.exp_names
    label_dims = train_sampler.label_dims

    # the number of labels is only the "positive" behaviors. The
    # do nothing label needs to be included.
    label_mat = numpy.zeros((experiments.size, label_dims + 1))
    vid_lengths = numpy.zeros((experiments.size,))
    train_sampler.reset()

    for i in range(train_sampler.num_batch):
        blob = train_sampler.get_minibatch()
        # the seq_len is 1 plus the last 1 entry in the mask.
        seq_len = numpy.max(numpy.argwhere(blob["masks"])) + 1

        # for each label/behavior, compute the number of appearances
        for j in range(label_dims):
            label_mat[i, j] = blob["labels"][:, 0, j].sum()

        label_mat[i, -1] =\
            seq_len - blob["labels"].sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = seq_len

    # label_counts = label_mat.sum(axis=0)

    label_weight = 1.0 / numpy.mean(label_mat, axis=0)
    # protect from div 0 issues.
    label_weight[numpy.isinf(label_weight)] = 0

    return label_weight


def _compute_means(opts, train_sampler):
    """Go over the features and compute the mean and variance."""
    # exp_names = train_data["exp_names"].value
    means = []
    stds = []
    if opts["flags"].normalize is True:
        running_stats = []
        for dim in train_sampler.feat_dims:
            running_stats.append(
                RunningStats(dim)
            )
        # loop over the experiments
        train_sampler.reset()

        for i in range(train_sampler.num_batch):
            # loop over the keys
            blob = train_sampler.get_minibatch()
            for j in range(len(blob["features"])):
                seq_len = numpy.max(numpy.argwhere(blob["masks"])) + 1
                feats = blob["features"][j]
                temp_feat = feats[:seq_len, 0, :]

                running_stats[j].add_data(
                    temp_feat
                )

        for i in range(len(opts["flags"].feat_keys)):
            means.append(running_stats[i].mean)
            stds.append(running_stats[i].compute_std())
    else:
        # for key in opts["flags"].feat_keys:
        for feat_dim in train_sampler.feat_dims():
            # temp_feat = train_data["exps"][exp_names[0]][key].value
            mean = numpy.zeros((feat_dim, ))
            std = numpy.ones((feat_dim, ))
            means.append(mean)
            stds.append(std)

    return means, stds


def _get_hidden(opts, network):
    if opts["flags"].cuda_device >= 0:
        use_cuda = True
    else:
        use_cuda = False

    hidden = network.init_hidden(
        opts["flags"].mini_batch,
        use_cuda=use_cuda)
    return hidden


def smooth_data(opts, org_labels):
    """Apply gaussian smoothing to the labels."""
    # -O -20 -10 -5 0 5 10 20 -d -4 -3 -2 -1 1 2 3 4 -w 19 -s 2
    smooth_window = 19
    smooth_std = 2

    if smooth_window == 0:
        return org_labels

    # org_labels = labels
    labels = numpy.zeros(org_labels.shape, dtype="float32")
    # loop over the columns and convolve
    conv_filter = signal.gaussian(smooth_window, std=smooth_std)
    # print conv_filter.shape
    # print labels.shape
    for i in range(labels.shape[1]):
        for j in range(labels.shape[2]):
            labels[:, i, j] = numpy.convolve(
                org_labels[:, i, j], conv_filter, 'same')
        # labels[:, i] = org_labels[:, i]
    # scale the labels a bit
    # labels = labels * 0.9
    # labels = labels + 0.01
    return labels


def _train_epoch(opts, step, network, optimizer, sampler, label_weights, means, stds):
    """Train one epoch."""
    if sampler.batch_idx.empty():
        sampler.reset()
    network.train()
    blob = sampler.get_minibatch()
    blob = sampler.get_minibatch()
    criterion = torch.nn.MSELoss().cuda()
    for i in range(100):
    #for i in range(sampler.num_batch):
        # blob = sampler.get_minibatch()
        hidden = _get_hidden(opts, network)

        inputs = []
        for j in range(len(blob["features"])):
            temp = ((blob["features"][j] - means[j])/stds[j]).astype("float32")
            inputs.append(torch.tensor(temp, requires_grad=True).cuda())
        labels = torch.tensor(blob["labels"], requires_grad=False).cuda()
        mask = torch.tensor(blob["masks"], requires_grad=False).cuda()

        train_predict, update_hid = network(inputs, hidden)        

        conv_labels = smooth_data(opts, blob["labels"])

        # TP_weight, FP_weight, false_neg, false_pos = create_match_array(
        #     opts, train_predict, labels, label_weight[2])
        # cost = total_cost.mean()
        for j in range(sampler.label_dims):
            conv_labels[:, :, j] = conv_labels[:, :, j] * label_weights[j]

        mask = torch.tensor(blob["masks"], requires_grad=False).cuda()
        conv_labels = conv_labels * mask
        train_predict = train_predict * mask

        conv_labels[:, :, 0] = 0
        conv_labels[:, :, 2:] = 0
        conv_labels = torch.tensor(conv_labels, requires_grad=False).cuda()

        cost = criterion(train_predict, conv_labels)
        optimizer.zero_grad()
        cost.backward()
        print(cost)
    import pdb; pdb.set_trace()


def _train_network(opts, network, optimizer, samplers):
    """Calling this will train the network."""
    # figure out label weights
    label_weights = _get_label_weight(samplers[1])
    means, stds = _compute_means(opts, samplers[1])
    print("Beginning training...")
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH: %d, iteration: %d" % (i, step))
        # first train an epoch of the data.
        tic = time.time()
        network.train()
        step += _train_epoch(opts, step, network, optimizer, samplers[0], label_weights, means, stds)
        print("\tTrain Epoch Time: %f" % (time.time() - tic))
        # print("\tFinished epoch")

        # # next, if update iterations, then process the full dataset.
        # if i % opts['flags'].update_iterations == 0 and i != 0:
        #     print("\tProcessing all examples...")
        #     tic = time.time()
        #     network.eval()
        #     train_cost = _eval_network(
        #         opts, step, network, samplers[0], criterion, "train")
        #     test_cost = _eval_network(
        #         opts, step, network, samplers[1], criterion, "test")
        #     valid_cost = _eval_network(
        #         opts, step, network, samplers[2], criterion, "valid")


def _setup_templates(opts, data_files, label_names):
    """Setup templates in the output folder.

    Given an ouptut directory and a list of open h5 data files,
    _setup_templates will create an output folder for each of the
    experiment folders.

    Args:
      opts: Option dictionary, created by _setup_opts.
      data_files: List of h5 data file handles. List should be of
        length 3: [train, test, validation] (in that order).
      label_names: Names of the labels. Needed for creating output
        templates for each label.
    """
    sequences_helper.copy_templates(
        opts, data_files[0], "train", label_names)
    sequences_helper.copy_templates(
        opts, data_files[1], "test", label_names)
    if len(data_files) == 3:
        sequences_helper.copy_templates(
            opts, data_files[2], "valid", label_names)


def _setup_samplers(opts, data_files):
    """Creates samplers for training and evaluation.

    For each of the h5 data handles, create a sampler.

    Args:
      opts: Option dictionary.
      data_files: List of h5 data file handles. List should be of
        length 2 or more: [train, test, validation] (in that order).

    Returns:
      list: [train, train_eval, test_eval, valid_eval] samplers, may
        not have the validation sampler, if no validation data was
        provided.
    """
    # assume the order is: train, test, valid for the hdf5 file
    # handles.
    # for training, create two samplers. one for training and one for
    # evaluation. The evaluation sampler will provide a video at
    # a time.

    # First construct the feature key list. If there are splits to
    # process, then the feature key can be a sub dict field.
    if opts["flags"].split is not None and opts["flags"].split is not "":
        feat_keys = [
            [opts["flags"].split, feat_key]
            for feat_key in opts["flags"].feat_keys
        ]
    else:
        feat_keys = opts["flags"].feat_keys

    # this is the sampler at train time.
    training_sampler = HDF5Sampler(
        opts["rng"], data_files[0], opts["flags"].mini_batch,
        feat_keys, seq_len=opts["flags"].seq_len,
        use_pool=False)

    # next create the evaluation samplers.
    samplers = [training_sampler]
    for data_file in data_files:
        # these samplers have batch size 1 and no random numnber
        # generator
        sampler = HDF5Sampler(None, data_file, 1, feat_keys,
                              seq_len=opts["flags"].seq_len,
                              use_pool=False)
        samplers.append(sampler)

    return samplers


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the number iterations per epoch.
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        num_train_vids = len(train_data["exp_names"])
        iter_per_epoch = numpy.ceil(
            1.0 * num_train_vids / opts["flags"].mini_batch)

        iter_per_epoch = int(iter_per_epoch)
        opts["flags"].iter_per_epoch = iter_per_epoch
        opts["flags"].total_iterations =\
            iter_per_epoch * opts["flags"].total_epochs

    return opts


def _init_network(opts, samplers):
    """Initialize the network."""
    feat_dims = samplers[0].feat_dims
    # label_dims = samplers[0].label_dims
    label_dims = 2

    if opts["flags"].arch == "concat":
        network = hantman_hungarian.HantmanHungarianConcat(
            input_dims=feat_dims,
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=label_dims
        )
        # network = torch.nn.DataParallel(network, device_ids=[0, 2])
    elif opts["flags"].arch == "sum":
        network = hantman_hungarian.HantmanHungarianSum(
            input_dims=feat_dims,
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=label_dims
        )
    else:
        network = hantman_hungarian.HantmanHungarianBidirConcat(
            input_dims=feat_dims,
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=label_dims
        )
    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    if opts["flags"].cuda_device != -1:
        network.cuda()

    return network, optimizer


def _load_hdfs(opts):
    """Load hdf5 files."""
    data_files = []
    data_files.append(h5py.File(opts["flags"].train_file, "r"))
    data_files.append(h5py.File(opts["flags"].test_file, "r"))

    if opts["flags"].valid_file is not None:
        data_files.append(h5py.File(opts["flags"].valid_file, "r"))

    return data_files


def main(argv):
    # setup options
    opts = _setup_opts(argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    # load data
    # timing here?
    tic = time.time()
    data_files = _load_hdfs(opts)
    # setup the templates for plotting
    _setup_templates(opts, data_files, g_label_names)
    samplers = _setup_samplers(opts, data_files)
    network, optimizer = _init_network(opts, samplers)
    _train_network(opts, network, optimizer, samplers)
    import pdb; pdb.set_trace()
    print(time.time() - tic)
    # close hdf5s
    for data_file in data_files:
        data_file.close()


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
