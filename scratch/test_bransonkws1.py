"""Mouse behavior spike classification."""
import gflags
import sys
import helpers.arg_parsing as arg_parsing
import numpy as np
import models.hantman_multi_lstm as hantman_multi_lstm

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
    network = hantman_multi_lstm.create_network(
        opts, num_input, num_classes)

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
    seq_len = opts["flags"].hantman_seq_length
    mini_batch = opts["flags"].hantman_mini_batch
    data = np.random.randn(seq_len, mini_batch, 1000).astype("float32")
    labels = np.random.randn(seq_len, mini_batch, 10).astype("float32")
    print seq_len
    print mini_batch
    print "STARTING FAKE TRAINING"
    for i in range(1000):
        print network["backprop"](data, labels)
        # print network["cost"](data, labels)[0]
