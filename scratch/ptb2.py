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
import numpy as np
import time
from docopt import docopt
import torch
import torch.nn as nn
import layers.lstm as lstm
import models.ptb_pytorch as ptb_pytorch
import models.ptb_model as ptb_model
from torch.autograd import Variable
# import theano

rng = np.random.RandomState()
eps = np.finfo(np.float32).eps


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def setup_ptb_opts(config_type):
    """Setup the options for the ptb experiment."""
    opts = dict()
    opts["rng"] = rng
    opts["eps"] = eps
    opts["data_path"] = "data"

    if config_type == "small":
        # default options are similar to the small configuration from the
        # google tensorflow penn tree bank rnn example.
        opts["scale"] = 0.1
        # opts["learning_rate"] = 1.0
        opts["learning_rate"] = 1.0
        opts["max_grad_norm"] = 5
        opts["num_lstm_layers"] = 2
        opts["seq_length"] = 20
        opts["num_hidden"] = 200
        opts["init_epoch_lr"] = 4
        opts["max_epochs"] = 13
        opts["keep_prob"] = 1.0
        opts["lr_decay"] = 0.5
        opts["batch_size"] = 20
        opts["vocab_size"] = 10000
        opts["optim"] = "sgd"
        opts["with_layer_norm"] = False
        opts["weight_type"] = "tied"
    elif config_type == "medium":
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


def _run_epoch(opts, is_verbose, data, is_train, model, model2, criterion):
    """Run one epoch of the model."""
    iters = 0
    costs = 0
    init_vals = lstm.create_initial_hidden_cell(
        opts["num_lstm_layers"], opts["batch_size"], opts["num_hidden"])
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

    hidden = model2.init_hidden(opts["batch_size"])
    start_time = time.time()
    lr = 1.0
    for step, (x, y) in enumerate(
        read_ptb.ptb_iterator(
            data, opts["batch_size"], opts["seq_length"])):
        # predict = model["predict"](data, y, *init_vals)
        x = x.T
        y = y.T
        output1 = _model_func(x, y, init_vals)
        x = torch.from_numpy(x.astype("int64"))
        x = Variable(x, volatile=False).cuda()
        y = torch.from_numpy(y.astype("int64")).contiguous().view(-1)
        y = Variable(y).cuda()

        hidden = repackage_hidden(hidden)
        model2.zero_grad()
        output, hidden = model2(x, hidden)
        loss = criterion(output.view(-1, opts["num_classes"]), y)
        loss.backward()
        import pdb; pdb.set_trace()
        # `clip_grad_norm` helps prevent the exploding gradient problem in
        # # RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model2.parameters(), 5)
        for p in model2.parameters():
            p.data.add_(-lr, p.grad.data)
        # import pdb; pdb.set_trace()
        cost = loss
        costs += cost.data[0]
        iters += opts["seq_length"]
        # output contains all h values, take on the end of the sequence
        # values.
        # init_vals = [seq[-1] for seq in output[1:]]
        # if step == 4:

        if is_verbose and step % (epoch_size // 10) == 10:
            # import pdb; pdb.set_trace()
            print "\t%.3f: perplexity: %.3f, learning rate: %f, wps: %f" %\
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                 lr,
                 iters * opts["batch_size"] / (time.time() - start_time))

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
    model2 = ptb_pytorch.RNNModel(
        "LSTM",
        opts["num_classes"],
        opts["num_hidden"],
        opts["num_hidden"],
        2,
        0.5,
        False
    )

    criterion = nn.CrossEntropyLoss()
    import pdb; pdb.set_trace()
    model2.cuda()
    # epoch training
    for i in range(opts["max_epochs"]):

        costs = _run_epoch(opts, True, train_data, 1, model, model2, criterion)
        print "Train Epoch %d finished: %f" % (i, costs)

        costs = _run_epoch(opts, False, valid_data, 0, model, model2, criterion)
        print "Valid Epoch %d finished: %f" % (i, costs)

    costs = _run_epoch(opts, False, test_data, 0, model, model2, criterion)
    print "Test Epoch %d finished: %f" % (i, costs)
