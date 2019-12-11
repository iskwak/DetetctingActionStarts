"""Mouse behavior spike classification."""
import gflags
import sys
import helpers.arg_parsing as arg_parsing
import numpy as np
import models.hantman_multi_lstm as hantman_multi_lstm
import theano
import theano.tensor as T
import optimizers.adam as adam

gflags.DEFINE_integer("total_iterations", 500000,
                      "Total number of iterations to trian the network.")
gflags.DEFINE_boolean("debug", False, "Debug flag, work with less videos.")
gflags.DEFINE_integer("update_iterations", None,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", None,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_boolean("threaded", True, "Threaded Data loadered.")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(hantman_multi_lstm)


def initialize_network_helper(opts):
    """Setup the network."""
    num_input = 1000
    num_classes = 10

    input = T.tensor4("input")
    # 256 x 256 -> 128 x 128 x 64
    # stride 2
    filter1 = theano.shared(np.asarray(
        opts["rng"].uniform(
            low=-1.0,
            high=1.0,
            size=(64, 3, 7, 7)),
            dtype=input.dtype), name ='filter1')
    out = T.nnet.conv2d(
        input, filter1, subsample=(2, 2), border_mode=(3, 3))

    filter2 = theano.shared(np.asarray(
        opts["rng"].uniform(
            low=-1.0,
            high=1.0,
            size=(2, 64, 1, 1)),
            dtype=input.dtype), name ='filter1')
    out = T.nnet.conv2d(
        out, filter2, subsample=(2, 2))

    # # create backprop
    label = T.tensor4("label")
    cost = T.sum(T.sqr(out - label))
    grad_ops = T.grad(cost, [filter1, filter2])

    backprop, learning_rate, grad_vals, optim_state = adam.create_adam_updates(
        [filter1, filter2], grad_ops, [input, label], cost)

    # # # batch, channels, rows, cols
    # moo = theano.function([input], out)
    # cow = np.zeros((100, 3, 256, 256), dtype="float32")
    # import pdb; pdb.set_trace()
    # out = T.nnet.conv2d(out, filter2, subsample=(2, 2))
    network = dict()
    network["backprop"] = backprop

    return network

if __name__ == "__main__":
    print sys.argv
    # opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
    FLAGS = gflags.FLAGS

    opts = dict()
    opts["eps"] = np.finfo(np.float32).eps
    opts["argv"] = sys.argv
    opts["rng"] = np.random.RandomState(123)
    # parse the inputs
    FLAGS(sys.argv)
    if arg_parsing.check_help(FLAGS) is True:
        sys.exit()

    opts["flags"] = FLAGS
    # load the network here
    network = initialize_network_helper(opts)

    # create some weird data
    # seq_len = opts["flags"].hantman_seq_length
    # mini_batch = opts["flags"].hantman_mini_batch
    # data = np.random.randn(seq_len, mini_batch, 1000).astype("float32")
    # labels = np.random.randn(seq_len, mini_batch, 10).astype("float32")
    mini_batch = opts["flags"].hantman_mini_batch

    data = np.random.randn(mini_batch, 3, 256, 256).astype("float32")
    labels = np.random.randn(mini_batch, 2, 64, 64).astype("float32")
    print "STARTING FAKE TRAINING"
    for i in range(1000):
        print network["backprop"](data, labels)
        # print network["cost"](data, labels)[0]
