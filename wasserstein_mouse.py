"""Mouse behavior spike classification."""
from __future__ import print_function, division
# import os
import time
import sys
import gflags
import numpy as np

import h5py
import helpers.paths as paths
import helpers.arg_parsing as arg_parsing
import helpers.videosampler
from helpers.videosampler import HDF5Sampler
import helpers.sequences_helper as sequences_helper

import train
import helpers.post_processing as post_processing
# import models.hantman_hungarian as hantman_hungarian
from models import hantman_hungarian
import flags.lstm_flags
import flags.cuda_flags
import torch
import os
# from helpers.RunningStats import RunningStats
# from scipy import signal

gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")
gflags.DEFINE_string("loss", "mse", "Loss to use for training.")

gflags.ADOPT_module_key_flags(helpers.videosampler)
gflags.ADOPT_module_key_flags(hantman_hungarian)
gflags.ADOPT_module_key_flags(flags.lstm_flags)
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)
gflags.ADOPT_module_key_flags(train)


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the feature key list. The hdf5 sampler expects each feature key
    # to be a list of sub fields. Allowing the hdf5 sampler to traverse
    # tree like hdf5 files. This isn't needed in this version of the
    # processing, so just wrap each element as a 1 item list.
    opts["flags"].feat_keys = [
        [feat_key] for feat_key in opts["flags"].feat_keys
    ]

    return opts


def run_training(opts, train_data, test_data, valid_data):
    """Setup sampler/output space and then run the network training.

    Given opts, and hdf5 data handles, setup the network training, then
    run the training.
    """
    timing_logname = os.path.join(opts["flags"].out_dir, "timing.csv")
    with open(timing_logname, "w") as fid:
        full_tic = time.time()
        fid.write("phase,timing\n")
        label_names = train_data["label_names"].value
        if valid_data is not None:
            data_files = [train_data, test_data, valid_data]
        else:
            data_files = [train_data, test_data]

        # setup output space.
        tic = time.time()
        _setup_templates(opts, data_files, label_names)
        toc = time.time()
        fid.write("output space,%f\n" % (toc - tic))
        print("Setup output space: %f" % (toc - tic))

        # create samplers for training/testing/validation.
        tic = time.time()
        samplers = _setup_samplers(opts, data_files)
        toc = time.time()
        fid.write("samplers,%f\n" % (toc - tic))
        print("Setup samplers: %f" % (toc - tic))

        # compute the label weighting.
        # train.get_label_weight(opts, train_data)
        print("getting label weights.")
        tic = time.time()
        if opts["flags"].reweight is True:
            label_weight = train.get_label_weight(samplers[1])
        else:
            label_weight = [1 for i in range(samplers[0].label_dims)]
        toc = time.time()
        fid.write("label weight,%f\n" % (toc - tic))
        print("Get label weights: %f" % (toc - tic))

        # create the network and optimizer.
        tic = time.time()
        network, optimizer, criterion = _init_network(opts, samplers, label_weight)
        toc = time.time()
        fid.write("network setup,%f\n" % (toc - tic))
        print("Setup network: %f" % (toc - tic))
        fid.flush()

        train.train_lstm(opts, network, optimizer, criterion, samplers, fid)
        toc = time.time()
        fid.write("full timing,%f\n" % (toc - full_tic))


def _init_network(opts, samplers, label_weights):
    """Initialize the network."""
    feat_dims = samplers[0].feat_dims
    label_dims = samplers[0].label_dims

    # compute the number of iterations per epoch.
    num_exp = len(samplers[0].exp_names)
    # iter_per_epoch =\
    #     np.ceil(1.0 * num_exp / opts["flags"].mini_batch)
    iter_per_epoch = 1.0 * num_exp / opts["flags"].mini_batch
    opts["flags"].perframe_decay_step = iter_per_epoch * opts["flags"].perframe_decay_step
    opts["flags"].iter_per_epoch = iter_per_epoch
    # import pdb; pdb.set_trace()

    # initialize the network
    if opts["flags"].hantman_arch == "concat":
        network = hantman_hungarian.HantmanHungarianConcat(
            input_dims=feat_dims,
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=label_dims
        )
    elif opts["flags"].hantman_arch == "sum":
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
    # optimizer = torch.optim.Adam(
    #     network.parameters(), lr=opts["flags"].learning_rate, weight_decay=0.00001)

    # next the criterion
    if opts["flags"].loss == "mse":
        # criterion = torch.nn.MSELoss(size_average=False).cuda()
        temp = torch.nn.MSELoss(size_average=False)
        if opts["flags"].cuda_device != -1:
            temp.cuda()

        def criterion(step, y, yhat, pos_mask, neg_mask, frame_mask):
            return temp(y, yhat)
        # criterion = lambda y, yhat, pos_mask, neg_mask:\
        #     temp(y, yhat)
    elif opts["flags"].loss == "weighted_mse":
        criterion = construct_weigthed_mse(opts, label_weights)
    elif opts["flags"].loss == "hungarian":
        criterion = construct_hungarian(opts, label_weights)
    elif opts["flags"].loss == "wasserstein":
        criterion = construct_wasserstein(opts, label_weights)

    if opts["flags"].cuda_device != -1:
        network.cuda()
    # import pdb; pdb.set_trace()
    return network, optimizer, criterion


def construct_wasserstein(opts, label_weights):
    # first split up label_weights
    neg_weight = torch.tensor(label_weights[-1], requires_grad=False).float()
    pos_weight = torch.tensor(label_weights[:-1], requires_grad=False).float()

    if opts["flags"].cuda_device != -1:
        neg_weight = neg_weight.cuda()
        pos_weight = pos_weight.cuda()

    def loss_fn(step, y, y_conv, yhat, pos_mask, neg_mask, frame_mask):
        # construct the wasserstein-1 in 1d.
        perframe_cost = wasserstein_perframe(
            y_conv, yhat, pos_mask, neg_mask, pos_weight, neg_weight)
        perframe_cost = perframe_cost.sum(0).sum(1)

        # The perframe_cost is the squared difference between each entry.
        # To get the MSE, calculate the mean
        # wasserstein_cost = wasserstein(opts, step, y, yhat, frame_mask)
        wasserstein_cost = wasserstein(opts, step, y_conv, yhat, frame_mask)
        # wasserstein_cost isn't collapsed yet. Each entry only has the
        # difference in the cumulative sums. Collapse to create the
        # wasserstein cost for each behavior.
        wasserstein_cost = wasserstein_cost.sum(0)
        # for the cost of the sequence, sum across behaviors.
        # wasserstein_cost = (pos_weight * wasserstein_cost).sum(1)
        wasserstein_cost = (pos_weight * wasserstein_cost).mean(1)

        perframe_lambda = hantman_hungarian.get_perframe_weight(
            opts, opts["flags"].hantman_perframe_weight, step)

        perframe_lambda = torch.autograd.Variable(
            torch.Tensor([float(perframe_lambda)]),
            requires_grad=False).cuda().expand(perframe_cost.size())

        # the goal of the wasserstein loss is to compute the difference in the
        # cumulative cdf's.
        # Let CDF_i(y) = \sum_{j=1}^i y_i
        # \sum_{i=1}^seq_len |CDF_i(Y) - CDF_i(\hat{Y})|
        # however this is the wasserstein loss for one behavior.
        # In order to calculate the total loss of all behaviors we add another
        # summations.
        # \sum_{c=1}^C \sum_{i=1}^seq_len |CDF_i(Y_c) - CDF_i(\hat{Y_c})|
        # To add weighting between types of classes, we can apply weighting to
        # the wasserstein loss of that class.
        # W(Y) = \sum_{c=1}^C w_c \sum_{i=1}^seq_len |CDF_i(Y_c) - CDF_i(\hat{Y_c})|
        # where Y is a matrix of sequence length x number of classes.
        # Note, in code we have wasserstein_cost as

        # Next combine this with the perframe loss.
        combined_loss =\
            perframe_lambda * perframe_cost +\
            (1 -  perframe_lambda) * wasserstein_cost

        # \frac{1}{batch} \sum_batch \sum_class class_weight * \sum_seq CDF diff
        # next apply mean on the batch
        combined_loss = combined_loss.mean()

        return combined_loss

    return loss_fn


def wasserstein_perframe(y, yhat, pos_mask, neg_mask, pos_weight, neg_weight):
    """Get the perframe wasserstein loss (squared error in this case)."""
    squared_error = (y - yhat) * (y - yhat)

    # expand out the pos_weight
    seq_len = pos_mask.shape[0]
    mini_batch = pos_mask.shape[1]
    pos_weight = pos_weight.repeat([seq_len, mini_batch, 1])

    # assumes that neg_mask intersection pos_mask is empty.
    perframe_cost =\
        squared_error * (pos_weight * pos_mask + neg_weight * neg_mask)
    return perframe_cost


def wasserstein(opts, step, y, yhat, frame_mask):
    eps = torch.from_numpy(np.asarray([opts["eps"]])).cuda()

    # y_sum = y.sum(0).detach()
    temp_y = (y + eps) * frame_mask
    y_sum = torch.tensor(temp_y.sum(0))
    # weird mask... but cause of the division. add it to things we
    # know have a 0 in the numerator.
    mask = torch.tensor(y_sum == 0, dtype=torch.float32, requires_grad=False).cuda()
    y = temp_y / (y_sum + eps)

    # yhat_sum = yhat.sum(0).detach()
    temp_yhat = yhat * frame_mask
    yhat_sum = torch.tensor(temp_yhat.sum(0))
    mask = torch.tensor(yhat_sum == 0, dtype=torch.float32, requires_grad=False).cuda()
    yhat = temp_yhat / (yhat_sum + eps)

    cdf_y = torch.cumsum(y, dim=0)
    cdf_yhat = torch.cumsum(yhat, dim=0)

    # wasser_dist = torch.abs(cdf_y - cdf_yhat) * frame_mask
    wasser_dist = torch.pow(cdf_y - cdf_yhat, 2) * frame_mask
    # wasser_dist2 = torch.pow((1 - cdf_y) - (1 - cdf_yhat), 2) * frame_mask
    # if step > 40:
    #     import pdb; pdb.set_trace()

    return wasser_dist


def construct_hungarian(opts, label_weights):
    # first split up label_weights
    neg_weight = torch.tensor(label_weights[-1], requires_grad=False).float()
    pos_weight = torch.tensor(label_weights[:-1], requires_grad=False).float()

    if opts["flags"].cuda_device != -1:
        neg_weight = neg_weight.cuda()
        pos_weight = pos_weight.cuda()

    def loss_fn(step, y, y_conv, yhat, pos_mask, neg_mask, frame_mask):
        return hungarian_loss(opts, step, y_conv, yhat, pos_mask, neg_mask, frame_mask, pos_weight, neg_weight)

    return loss_fn


def hungarian_loss(opts, step, y, yhat, pos_mask, neg_mask, mask, pos_weight, neg_weight):
    """Hungarian loss"""
    # figure out the matches.
    TP_weight, FP_weight, num_false_neg, num_false_pos = create_match_array(
        opts, yhat, y, pos_weight, neg_weight)

    seq_len = pos_mask.shape[0]
    mini_batch = pos_mask.shape[1]
    pos_weight = pos_weight.repeat([seq_len, mini_batch, 1])

    pos_mask, neg_mask = hantman_hungarian.create_pos_neg_masks(y, pos_weight, neg_weight)
    perframe_cost = hantman_hungarian.perframe_loss(yhat, mask, y, pos_mask, neg_mask)
    tp_cost, fp_cost, fn_cost = hantman_hungarian.structured_loss(
        yhat, mask, TP_weight, FP_weight, num_false_neg)

    total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost =\
        hantman_hungarian.combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost)
    cost = total_cost.mean()

    return cost


def create_match_array(opts, net_out, org_labels, pos_weight, neg_weight):
    """Create the match array."""
    val_threshold = 0.7
    # frame_threshold = [5, 15, 15, 20, 30, 30]
    frame_threshold = [10, 10, 10, 10, 10, 10]
    # frame_threshold = 10
    # y_org = org_labels
    y_org = org_labels.data.cpu().numpy()

    COST_FP = 20
    # COST_FN = 20
    net_out = net_out.data.cpu().numpy()
    num_frames, num_vids, num_classes = net_out.shape
    TP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    FP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    num_false_neg = []
    num_false_pos = []
    for i in range(num_vids):
        temp_false_neg = 0
        temp_false_pos = 0
        for j in range(num_classes):
            processed, max_vals = post_processing.nonmax_suppress(
                net_out[:, i, j], val_threshold)
            processed = processed.reshape((processed.shape[0], 1))
            data = np.zeros((len(processed), 3), dtype="float32")
            data[:, 0] = list(range(len(processed)))
            data[:, 1] = processed[:, 0]
            data[:, 2] = y_org[:, i, j]
            # if opts["flags"].debug is True:
            #     import pdb; pdb.set_trace()
            # after suppression, apply hungarian.
            labelled = np.argwhere(y_org[:, i, j] == 1)
            labelled = labelled.flatten().tolist()
            num_labelled = len(labelled)
            dist_mat = post_processing.create_frame_dists(
                data, max_vals, labelled)
            rows, cols, dist_mat = post_processing.apply_hungarian(
                dist_mat, frame_threshold[j])

            # missed classifications
            # false_neg = len(labelled) - len(
            #     [k for k in range(len(max_vals)) if cols[k] < len(labelled)])
            # num_false_neg += false_neg
            # temp_false_neg += false_neg

            num_matched = 0
            for pos in range(len(max_vals)):
                ref_idx = max_vals[pos]
                # if cols[pos] < len(labelled):
                row_idx = rows[pos]
                col_idx = cols[pos]
                if col_idx < len(labelled) and\
                        dist_mat[row_idx, col_idx] < frame_threshold[j]:
                    # True positive
                    label_idx = labelled[cols[pos]]
                    # TP_weight[ref_idx, i, j] = np.abs(ref_idx - label_idx)
                    TP_weight[ref_idx, i, j] = 10 - np.abs(ref_idx - label_idx)
                    # import pdb; pdb.set_trace()
                    # if we are reweighting based off label rariety
                    if opts["flags"].reweight is True:
                        # TP_weight[ref_idx, i, j] =\
                        #     TP_weight[ref_idx, i, j] * pos_weigth[j]
                        TP_weight[ref_idx, i, j] =\
                            TP_weight[ref_idx, i, j] * pos_weight[j]
                    TP_weight[ref_idx, i, j] = TP_weight[ref_idx, i, j] / 10
                    num_matched += 1
                else:
                    # False positive
                    FP_weight[ref_idx, i, j] = opts["flags"].hantman_fp
                    if opts["flags"].reweight is True:
                        FP_weight[ref_idx, i, j] =\
                            FP_weight[ref_idx, i, j] * pos_weight[j]
                    temp_false_pos += 1

            temp_false_neg += num_labelled - num_matched
        num_false_neg.append(temp_false_neg)
        num_false_pos.append(temp_false_pos)

    num_false_neg = np.asarray(num_false_neg).astype("float32")

    return TP_weight, FP_weight, num_false_neg, num_false_pos


def construct_weigthed_mse(opts, label_weights):
    """Helper function to create the weigthed mse function.

    Create a weighted mse loss function.
    """
    # first split up label_weights
    neg_weight = torch.tensor(label_weights[-1], requires_grad=False).float()
    pos_weight = torch.tensor(label_weights[:-1], requires_grad=False).float()

    if opts["flags"].cuda_device != -1:
        neg_weight = neg_weight.cuda()
        pos_weight = pos_weight.cuda()

    # loss_fn = lambda y, yhat, pos_mask, neg_mask:\
    #     weighted_mse(y, yhat, pos_mask, neg_mask, pos_weight, neg_weight)
    def loss_fn(step, y, y_conv, yhat, pos_mask, neg_mask, frame_mask):
        return weighted_mse(y_conv, yhat, pos_mask, neg_mask, pos_weight, neg_weight)

    return loss_fn


def weighted_mse(y, yhat, pos_mask, neg_mask, pos_weight, neg_weight):
    """Weighted MSE Loss Function.

    Given labels, predictions, positive mask, negative mask, and label weights,
    this function will compute a weighted MSE.
    """
    mse = (y - yhat) * (y - yhat)

    # expand out the pos_weight
    seq_len = pos_mask.shape[0]
    mini_batch = pos_mask.shape[1]
    pos_weight = pos_weight.repeat([seq_len, mini_batch, 1])

    # assumes that neg_mask intersection pos_mask is empty.
    weighted_mse = mse * (pos_weight * pos_mask + neg_weight * neg_mask)

    return weighted_mse.sum()


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
    # if opts["flags"].split is not None and opts["flags"].split is not "":
    #     feat_keys = [
    #         [opts["flags"].split, feat_key]
    #         for feat_key in opts["flags"].feat_keys
    #     ]
    # else:
    feat_keys = opts["flags"].feat_keys

    # this is the sampler at train time.
    training_sampler = HDF5Sampler(
        opts["rng"], data_files[0], opts["flags"].mini_batch,
        feat_keys, seq_len=opts["flags"].seq_len,
        use_pool=opts["flags"].use_pool,
        max_workers=opts["flags"].max_workers,
        gpu_id=opts["flags"].cuda_device)

    # next create the evaluation samplers.
    samplers = [training_sampler]
    for data_file in data_files:
        # these samplers have batch size 1 and no random numnber
        # generator
        sampler = HDF5Sampler(None, data_file, 1, feat_keys,
                              seq_len=opts["flags"].seq_len,
                              use_pool=False, gpu_id=opts["flags"].cuda_device)
        samplers.append(sampler)

    return samplers


def main(argv):
    print(argv)
    opts = _setup_opts(argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    full_tic = time.time()
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            if opts["flags"].valid_file is not None:
                with h5py.File(opts["flags"].valid_file, "r") as valid_data:
                    run_training(opts, train_data, test_data, valid_data)
            else:
                run_training(opts, train_data, test_data, None)
    print("Training took: %d\n" % (time.time() - full_tic))


if __name__ == "__main__":
    main(sys.argv)
