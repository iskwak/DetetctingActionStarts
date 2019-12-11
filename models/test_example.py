"""Test the example LSTM model."""
from __future__ import print_function, division
import sys
import os
import numpy as np

import example_lstm
import torch
import h5py
import gflags

gflags.DEFINE_integer("cuda_device", 0, "Which CUDA device to use, -1 for cpu.")

# seq_len = 100
batch_size = 20
output_dim = 6
rng = np.random.RandomState(123)
num_iters = 1000
rng = np.random.RandomState(123)


def _init_network(FLAGS):
    """Setup the network."""
    network = example_lstm.HantmanHungarianConcat(
        input_dims=[4096, 4096],
        hidden_dim=64,
        output_dim=output_dim
    )

    # create the optimizer too.
    # the torch.nn.module library does some black magic to properly setup
    # the network.parameters call.
    optimizer = torch.optim.Adam(
        network.parameters(), lr=0.001)

    # setup the criterion
    criterion = torch.nn.MSELoss()

    if FLAGS.cuda_device != -1:
        network.cuda()
        criterion.cuda()

    return network, optimizer, criterion


def _get_minibatch():
    """Create a sample minibatch."""
    # batch should be of form (seq_len, batch_size, feature_dim)
    # there are 6 hdf5 files in the test_data directory. Randomly choose
    # two of them.
    exp_files = os.listdir("test_data")
    permuted = rng.permutation(len(exp_files))

    all_feats = [
        np.zeros((1000, batch_size, 4096)),
        np.zeros((1000, batch_size, 4096))
    ]
    all_labels = np.zeros((1000, batch_size, output_dim))
    for i in range(batch_size):
        fname = os.path.join("test_data", exp_files[permuted[1]])
        with h5py.File(fname, "r") as h5data:
            curr_feats = h5data["img_side_norm"].value
            max_idx = min(curr_feats.shape[0], 1000)
            all_feats[0][:max_idx, i, :] = curr_feats[:, 0, :]

            curr_feats = h5data["img_front_norm"].value
            all_feats[1][:max_idx, i, :] = curr_feats[:, 0, :]

            curr_labels = h5data["labels"].value
            all_labels[:max_idx, i, :] = curr_labels[:, 0, :]

    return all_feats, all_labels


def main(argv):
    FLAGS = gflags.FLAGS
    FLAGS(argv)
    if FLAGS.cuda_device != -1:
        torch.cuda.set_device(FLAGS.cuda_device)
    network, optimizer, criterion = _init_network(FLAGS)

    network = torch.nn.DataParallel(network, device_ids=[0, 2], dim=1)

    # Put the network in train mode. If I understand correctly, "unfreezes" the
    # weights and tells torch to do backwards computations while doing the
    # forward pass.
    network.train()

    for i in range(num_iters):
        # Make some fake data to overfit to.
        feats, labels = _get_minibatch()
        temp = list(network.modules())
        temp = temp[1]
        hidden = temp.init_hidden(FLAGS, batch_size)

        # convert to torch tensor
        feats = [
            torch.from_numpy(feats[0]).float(),
            torch.from_numpy(feats[1]).float()
        ]
        labels = torch.from_numpy(labels).float()
        # convert the numpy arrays to torch tensor variables
        # cuda-fy
        if FLAGS.cuda_device != -1:
            feats = [feats[0].cuda(), feats[1].cuda()]
            labels = labels.cuda()

        feats_var = [
            torch.autograd.Variable(feats[1], requires_grad=True),
            torch.autograd.Variable(feats[0], requires_grad=True),
        ]
        labels_var = torch.autograd.Variable(
            labels, requires_grad=False)

        # the inputs to the network is a feature list and hidden state tuple.
        # These need to be torch variables. Note: init_hidden has already made
        # the tuple into a variable.
        train_predict, update_hidden = network(feats_var, hidden)

        # next compute the loss.
        optimizer.zero_grad()
        loss = criterion(train_predict, labels_var)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Loss at %d iters: %f" % (i, loss.data[0]))
            # simulate diskwriting.
            with open("test_out/results.csv", "w") as f:
                for j in range(1000):
                    for k in range(6):
                        f.write("%f," % train_predict.data.cpu()[j, 0, k])
                    f.write("\n")
    print("finished training.")


if __name__ == "__main__":
    main(sys.argv)
