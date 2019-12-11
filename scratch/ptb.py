"""Ptb tests.

Usage:
      ptb.py [--config=<config_type | -c <config_type>]
      ptb.py (-h | --help)

Options:
    -h --help          Show this screen.
    -c <config_type>, --config=<config_type>   Network Config and this is all a
                                               test [default: medium].
"""
import helpers.read_ptb as read_ptb
import models.ptb_model2 as ptb_model
import numpy as np
import layers.lstm as lstm
import time
from docopt import docopt
# import theano

rng = np.random.RandomState()
eps = np.finfo(np.float32).eps


def setup_ptb_opts(config_type):
    """Setup the options for the ptb experiment."""
    opts = dict()
    opts["rng"] = rng
    opts["eps"] = eps
    opts["data_path"] = "data"

    if config_type == "small":
        # default options are similar to the small configuration from the
        # google tensorflow penn tree bank rnn example.
        # opts["scale"] = 0.1
        # # opts["learning_rate"] = 1.0
        # opts["learning_rate"] = 1.0
        # opts["max_grad_norm"] = 5
        # opts["num_lstm_layers"] = 2
        # opts["seq_length"] = 20
        # opts["num_hidden"] = 200
        # opts["init_epoch_lr"] = 4
        # opts["max_epochs"] = 13
        # opts["keep_prob"] = 1.0
        # opts["lr_decay"] = 0.5
        # opts["batch_size"] = 25
        # opts["vocab_size"] = 10000
        # opts["optim"] = "sgd"
        # opts["with_layer_norm"] = False
        # opts["weight_type"] = "tied"

        opts["lstm_scale"] = 0.1
        opts["scale"] = 0.1
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["lstm_layers"] = 2
        opts["seq_length"] = 20
        opts["lstm_hidden_dim"] = 200
        opts["lstm_input_dim"] = 200
        opts["init_epoch_lr"] = 4
        opts["max_epochs"] = 13
        opts["lstm_dropout_keep"] = 1.0
        opts["lr_decay"] = 0.5
        opts["batch_size"] = 25
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["lstm_weight_type"] = "tied"
        opts["lstm_weight_init"] = "uniform"
        opts["lstm_forget_bias"] = 0.0
    elif config_type == "medium":
        # medium example from tensorflow rnn example.
        opts["lstm_scale"] = 0.05
        opts["scale"] = 0.05
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["lstm_hidden_dim"] = 650
        opts["lstm_input_dim"] = 650
        opts["init_epoch_lr"] = 6
        opts["max_epochs"] = 39
        opts["lstm_dropout_keep"] = 0.5
        opts["lr_decay"] = 0.8
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["lstm_weight_type"] = "tied"
        opts["lstm_weight_init"] = "uniform"
        opts["lstm_forget_bias"] = 0.0
    elif config_type == "medium_ortho":
        # medium example from tensorflow rnn example.
        opts["scale"] = 0.05
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["num_hidden"] = 650
        opts["init_epoch_lr"] = 6
        opts["max_epochs"] = 39
        opts["keep_prob"] = 0.5
        opts["lr_decay"] = 0.8
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["weight_type"] = "tied"
    elif config_type == "medium_layer_norm":
        opts["scale"] = 0.05
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["num_hidden"] = 650
        opts["init_epoch_lr"] = 6
        opts["max_epochs"] = 39
        opts["keep_prob"] = 0.5
        opts["lr_decay"] = 0.8
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = True
        opts["weight_type"] = "split"
    elif config_type == "split_medium":
        opts["scale"] = 0.05
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["num_hidden"] = 650
        opts["init_epoch_lr"] = 6
        opts["max_epochs"] = 39
        opts["keep_prob"] = 0.5
        opts["lr_decay"] = 0.8
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["weight_type"] = "split"
    elif config_type == "adam_medium":
        opts["scale"] = 0.05
        opts["learning_rate"] = 0.01
        opts["max_grad_norm"] = 5
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["num_hidden"] = 650
        opts["init_epoch_lr"] = 6
        opts["max_epochs"] = 39
        opts["keep_prob"] = 0.5
        opts["lr_decay"] = 0.8
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "adam"
        opts["with_layer_norm"] = False
        opts["weight_type"] = "tied"
    elif config_type == "large":
        opts["scale"] = 0.04
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 10
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 35
        opts["num_hidden"] = 1600
        opts["init_epoch_lr"] = 14
        opts["max_epochs"] = 55
        opts["keep_prob"] = 0.35
        opts["lr_decay"] = 1 / 1.15
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["weight_type"] = "tied"
    else:
        raise NotImplemented

    return opts


def create_label_idx(y):
    """Create a label idx"""
    label_idx = np.zeros((y.size, 3), dtype="int32")
    row_idx = 0
    # for j in range(y.shape[1]):
    #     for i in range(y.shape[0]):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            label_idx[row_idx, :] = [i, j, y[i, j]]
            row_idx += 1

    return label_idx


def _run_epoch(opts, is_verbose, data, is_train, model):
    """Run one epoch of the model."""
    iters = 0
    costs = 0
    init_vals = lstm.create_initial_hidden_cell(
        opts["lstm_layers"], opts["batch_size"], opts["lstm_hidden_dim"])
    epoch_size =\
        ((len(train_data) // opts["batch_size"]) - 1) // opts["seq_length"]

    if is_train == 0:
        func = model["cost"]
    else:
        func = model["backprop"]

    # create a lambda function to help with the calls
    # if opts["keep_prob"] < 1.0:
    #     def _model_func(x, y, init_vals):
    #         return func(x, y, is_train, *init_vals)
    # else:
    def _model_func(x, y, init_vals):
        return func(x, y, *init_vals)

    start_time = time.time()
    for step, (x, y) in enumerate(
        read_ptb.ptb_iterator(
            data, opts["batch_size"], opts["seq_length"])):
        # predict = model["predict"](data, y, *init_vals)
        x = x.T
        y = y.T

        # moo = model["predict"](x, y, *init_vals)
        # label_idx = create_label_idx(y)
        # import pdb; pdb.set_trace()
        # output = _model_func(x, y, init_vals)
        label_idx = create_label_idx(y)
        # import pdb; pdb.set_trace()
        output = _model_func(x, label_idx, init_vals)

        cost = output[0]
        costs += cost
        iters += opts["seq_length"]
        # output contains all h values, take on the end of the sequence
        # values.
        init_vals = [seq[-1] for seq in output[1:]]

        if is_verbose and step % (epoch_size // 10) == 10:
            # import pdb; pdb.set_trace()
            print "\t%.3f: perplexity: %.3f, learning rate: %f, wps: %f" %\
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 model["learning_rate"].eval(),
                 iters * opts["batch_size"] / (time.time() - start_time))
            # moo = model["cost"](x, y, *init_vals)
            # label_idx = create_label_idx(y)
            # moo1 = model["debug1"](x, label_idx, *init_vals)
            # print np.sum(np.square(moo[0] - moo1[1]))

    return np.exp(costs / iters)


if __name__ == "__main__":
    # load the data
    # opts = default_opts()
    # opts = split_medium_opts()
    # opts = adam_medium_opts()
    # opts = medium_layer_norm_opts()
    arguments = docopt(__doc__, version="pubsub_daemon 1.0")
    print arguments
    opts = setup_ptb_opts(arguments["--config"])
    # run_daemon(arguments)

    train_data, valid_data, test_data, vocab = read_ptb.ptb_raw_data(
        data_path="data")
    opts["num_input"] = vocab
    opts["num_classes"] = vocab
    opts["num_iter_per_epoch"] =\
        len(train_data) // (opts["batch_size"] * opts["seq_length"])
    epoch_size =\
        ((len(train_data) // opts["batch_size"]) - 1) // opts["seq_length"]

    model = ptb_model.ptb_model(opts)

    # epoch training
    for i in range(opts["max_epochs"]):

        costs = _run_epoch(opts, True, train_data, 1, model)
        print "Train Epoch %d finished: %f" % (i, costs)

        costs = _run_epoch(opts, False, valid_data, 0, model)
        print "Valid Epoch %d finished: %f" % (i, costs)

    costs = _run_epoch(opts, False, test_data, 0, model)
    print "Test Epoch %d finished: %f" % (i, costs)
