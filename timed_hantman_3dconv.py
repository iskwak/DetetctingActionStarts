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
import flags.cuda_flags
import torch
from torch.autograd import Variable
from models.hantman_3dconv import Hantman3DConv
from helpers.hantman_sampler import HantmanSeqFrameSampler
from helpers.hantman_sampler import HantmanVideoFrameSampler

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


# global variables for timing
g_training = []
g_eval = []


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


def _init_network(opts):
    """Setup the network."""
    network = Hantman3DConv().cuda()
    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    criterion = torch.nn.NLLLoss()

    if opts["flags"].cuda_device != -1:
        network.cuda()
        criterion.cuda()

    return network, optimizer, criterion


def _train_epoch(opts, network, optimizer, criterion, sampler):

    for i in range(sampler.num_batch):
        cow = sampler.get_minibatch()
        cow[0] = cow[0].cuda()
        cow[1] = cow[1].cuda()
        cow[2] = cow[2].cuda()

        out = network(cow[:2])
        loss = criterion(out, cow[2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # import pdb; pdb.set_trace()
    print("moo")
    return sampler.num_batch


def _train_network(opts, network, optimizer, criterion, train_data, test_data,
                   sampler, train_eval, test_eval):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        tic = time.time()
        network.train()
        step += _train_epoch(opts, network, optimizer, criterion, sampler)
        print("\t%f" % (time.time() - tic))
        print("\tFinished epoch")
        print("\tProcessing all examples...")

        if i % 20 == 0:
            tic = time.time()
            network.eval()
            train_cost = _eval_network(
                opts, step, network, train_eval, criterion, "train")
            test_cost = _eval_network(
                opts, step, network, test_eval, criterion, "test")
            _log_outputs(opts, step, train_cost, test_cost)
            print("\teval time: %f" % (time.time() - tic))
        # network.eval()
        # _log_outputs(opts, step, network, label_weight)
        # round_tic = time.time()

        # save the network in its own folder in the networks folder
        out_dir = os.path.join(
            opts["flags"].out_dir, "networks", "%d" % step)
        paths.create_dir(out_dir)
        out_name = os.path.join(out_dir, "network.pt")
        torch.save(network.cpu().state_dict(), out_name)
        network.cuda()
        # hantman_hungarian_image.save_network(opts, network, out_name)
    print("Finished training.")


def _load_chunk(opts, inputs, frames):
    feat1 = torch.zeros(1, 1, 10, 224, 224)
    feat2 = torch.zeros(1, 1, 10, 224, 224)

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


def _create_chunks(opts, inputs, idx1, idx2):
    """Create the overlapping chunks."""
    # idx2 = 75
    # idx1 = 71
    num_batch = idx2 - idx1
    # img1 = torch.zeros(num_batch, 1, 10, 224, 224)
    # img2 = torch.zeros(num_batch, 1, 10, 224, 224)
    # labels = torch.zeros(num_batch)

    feat1_list = []
    feat2_list = []
    label_list = []
    for i in range(num_batch):
        curr_idx = i + idx1
        frames = range(curr_idx - 5, curr_idx + 5)
        temp1, temp2 = _load_chunk(opts, inputs, frames)
        feat1_list.append(temp1)
        feat2_list.append(temp2)

        temp_label = inputs[2][curr_idx, :].nonzero()
        if len(temp_label.size()) == 0:
            temp_label = 6
        else:
            temp_label = temp_label[0][0]
        label_list.append(temp_label)

    feat1 = torch.cat(feat1_list, dim=0)
    feat2 = torch.cat(feat2_list, dim=0)
    labels = torch.LongTensor(label_list)
    return feat1, feat2, labels


def _eval_network(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()

        num_frames = inputs[0].size()[0]
        chunk_len = 4
        num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
        predict = np.zeros((num_frames, 1, 6))
        # print(num_frames)
        for j in range(0, num_chunks):
            idx1 = j * chunk_len
            idx2 = min((j + 1) * chunk_len, num_frames)
            # print("\t%d" % idx2)
            feat1, feat2, labels = _create_chunks(opts, inputs, idx1, idx2)
            chunk = [
                Variable(feat1, requires_grad=True).cuda(),
                Variable(feat2, requires_grad=True).cuda(),
                Variable(labels, requires_grad=False).cuda()
            ]
            out = network([chunk[0], chunk[1]])
            # if len((labels != 6).nonzero().size()):
            #     import pdb; pdb.set_trace()
            temp = torch.exp(out.data)
            predict[idx1:idx2, 0, :] = temp[:, :6].cpu().numpy()

            loss = criterion(out, chunk[-1])
            total_loss += loss.data[0]
        exp_names = [inputs[3]]
        labels = inputs[2].cpu().numpy()
        labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        frames = [range(labels[0].shape[0])]
        sequences_helper.write_predictions2(out_dir, exp_names, predict, labels,
                                            None, frames)
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch)


def _log_outputs(opts, step, train_cost, test_cost):
    # apply post processing (hungarian matching and create cleaned outputs).
    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "train")
    train_dicts = post_processing.process_outputs(
        predict_dir, "")

    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "test")
    test_dicts = post_processing.process_outputs(
        predict_dir, "")

    # after applying the post processing,
    trainf, trainf_scores = compute_tpfp(opts, train_dicts)
    testf, testf_scores = compute_tpfp(opts, test_dicts)

    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss_f.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,train f1,test f1,"
                     "train lift,train hand,train grab,train supinate,"
                     "train mouth,train chew,"
                     "test lift,test hand,test grab,test supinate,"
                     "test mouth,test chew\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = ("%d,%f,%f,%f,%f,"
                      "%f,%f,%f,%f,"
                      "%f,%f,"
                      "%f,%f,%f,%f,"
                      "%f,%f\n")
        output_data =\
            [step, train_cost, test_cost, trainf, testf] +\
            trainf_scores + testf_scores
        output_data = tuple(output_data)
        # import pdb; pdb.set_trace()
        outfile.write(format_str % output_data)
    print("\tupdated...")


def compute_tpfp(opts, label_dicts):
    """Compute the precision recall information."""
    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    mean_f = 0
    for i in range(len(label_dicts)):
        tp = float(len(label_dicts[i]['dists']))
        fp = float(label_dicts[i]['fp'])
        fn = float(label_dicts[i]['fn'])
        precision = tp / (tp + fp + opts["eps"])
        recall = tp / (tp + fn + opts["eps"])
        f1_score =\
            2 * (precision * recall) / (precision + recall + opts["eps"])

        total_tp += tp
        total_fp += fp
        total_fn += fn
        # print "label: %s" % label_dicts[i]['label']
        # print "\tprecision: %f" % precision
        # print "\trecall: %f" % recall
        # print "\tfscore: %f" % f1_score
        # mean_f += f1_score
        f1_scores.append(f1_score)

    precision = total_tp / (total_tp + total_fp + opts["eps"])
    recall = total_tp / (total_tp + total_fn + opts["eps"])
    mean_f = 2 * (precision * recall) / (precision + recall + opts["eps"])
    # mean_f = (mean_f / len(label_dicts))
    # print "mean score: %f" % mean_f

    return mean_f, f1_scores


def main(argv):
    opts = _setup_opts(sys.argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            _copy_templates(opts, train_data, test_data)
            sampler = HantmanSeqFrameSampler(
                opts["rng"], train_data, opts["flags"].image_dir,
                opts["flags"].hantman_mini_batch, use_pool=False)

            train_eval = HantmanVideoFrameSampler(
                train_data, "/media/drive1/data/hantman_frames",
                use_pool=True, gpu_id=opts["flags"].cuda_device)
            test_eval = HantmanVideoFrameSampler(
                test_data, "/media/drive1/data/hantman_frames",
                use_pool=True, gpu_id=opts["flags"].cuda_device)

            network, optimizer, criterion = _init_network(opts)

            _train_network(
                opts, network, optimizer, criterion, train_data, test_data,
                sampler, train_eval, test_eval
            )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
