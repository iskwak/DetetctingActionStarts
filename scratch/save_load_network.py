"""Save/Load testing."""
import gflags
import helpers.paths as paths
import os
import h5py
import sys
import helpers.arg_parsing as arg_parsing
import numpy as np
from sklearn.externals import joblib
# import helpers.git_helper as git_helper
import helpers.hantman_mouse as hantman_mouse
import helpers.sequences_helper as sequences_helper
import models.hantman_multi_lstm as hantman_multi_lstm
# import time
import theano

gflags.DEFINE_string("out_dir1", None, "Output directory1 path.")
gflags.DEFINE_string("out_dir2", None, "Output directory2 path.")
gflags.DEFINE_string("filename", None, "Feature data filename (hdf5).")
gflags.DEFINE_string("image_dir", None, "Directory for images to symlink.")
gflags.DEFINE_integer("total_iterations", 500000,
                      "Total number of iterations to trian the network.")
gflags.DEFINE_boolean("debug", False, "Debug flag, work with less videos.")
gflags.DEFINE_integer("update_iterations", 1000,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("save_iterations", 10000,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.MarkFlagAsRequired("out_dir1")
gflags.MarkFlagAsRequired("out_dir2")
gflags.MarkFlagAsRequired("filename")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(hantman_multi_lstm)


def _get_frame_counts(h5_data, idx):
    """Get the number of frames for each video."""
    frame_counts = []
    for exp in idx:
        num_frames = h5_data["exps"][exp]["reduced"].value.shape
        frame_counts.append(range(num_frames[0]))
    return frame_counts


def _get_seq_mini_batch(opts, batch_id, h5_data, feat_idx, idx):
    """Get a mini-batch of data."""
    if "mini_batch_size" in opts.keys():
        batch_size = opts["mini_batch_size"]
    else:
        batch_size = opts["flags"].hantman_mini_batch

    start_idx = batch_id * batch_size
    end_idx = start_idx + batch_size
    if end_idx >= len(idx):
        # valid_idx = end_idx - len(idx)
        buf = [0 for i in range(end_idx - len(idx))]
        end_idx = len(idx) - 1
        sample_idx = np.asarray(range(start_idx, len(idx)) + buf,
                                dtype="int64")
        vid_idx = sample_idx
        sample_idx = sample_idx + feat_idx

        valid_idx = np.max(np.argwhere(vid_idx + 1 == len(idx))) + 1
        batch_id = -1
    else:
        sample_idx = np.asarray(range(start_idx, end_idx), dtype="int64")
        vid_idx = sample_idx
        sample_idx = sample_idx + feat_idx

        batch_id += 1
        valid_idx = len(sample_idx)

    if opts["flags"].hantman_givens:
        return [sample_idx], vid_idx, batch_id, valid_idx
    # else make the arrays
    feat_dims = h5_data["exps"][idx[0]]["reduced"].shape[2]

    all_feats = np.zeros(
        (1500, batch_size, feat_dims), dtype=theano.config.floatX)
    all_feats = all_feats - 5
    all_labels = np.zeros((1500, batch_size, 6), dtype=theano.config.floatX)

    for i in range(batch_size):
        temp_idx = idx[vid_idx[i]]

        # temp_idx = idx[i]
        feat = h5_data["exps"][temp_idx]["reduced"].value

        seq_len = feat.shape[0]
        all_feats[:seq_len, i, :] =\
            feat.reshape((feat.shape[0], feat.shape[2]))

        label = h5_data["exps"][temp_idx]["labels"].value
        all_labels[:seq_len, i, :] =\
            label.reshape((label.shape[0], label.shape[2]))

    return [all_feats, all_labels], vid_idx, batch_id, valid_idx


def _process_full_sequences(opts, network, h5_data, train_idx, test_idx):
    """Make per sequence predictions."""
    train_dir = os.path.join(opts["flags"].out_dir, "predictions", "train")
    test_dir = os.path.join(opts["flags"].out_dir, "predictions", "test")

    _predict_write(opts, train_dir, network, h5_data, 0, train_idx)
    _predict_write(opts, test_dir, network, h5_data, len(train_idx), test_idx)


def _predict_write(opts, out_dir, network, h5_data, feat_idx, vid_idx):
    """Predict and write sequence classifications."""
    batch_id = 0
    while batch_id != -1:
        inputs, idx, batch_id, valid_idx = _get_seq_mini_batch(
            opts, batch_id, h5_data, feat_idx, vid_idx)

        outputs = network["cost"](*inputs)
        predictions = outputs[1]

        # collect the labels
        labels = []
        frames = []
        for vid in vid_idx[idx]:
            labels.append(h5_data["exps"][vid]["labels"].value)
            frames.append(range(h5_data["exps"][vid]["labels"].shape[0]))

        idx = idx[:valid_idx]
        # print feat_idx
        # print idx
        # print inputs
        sequences_helper.write_predictions2(
            out_dir, vid_idx[idx], predictions, labels, frames)


def _copy_params(network):
    """Copy the params."""
    params = []
    learnable = network["params"]
    for param in learnable:
        params.append(np.copy(param.get_value()))
    return params


def train_network(opts, network, out_dir, h5_data, train_vids, test_vids):
    """Train the network."""
    # setup the data samplers
    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, train_vids)
    train_sampler = sampler

    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, test_vids, len(train_vids))
    test_sampler = sampler

    t = network["lr_update"]["params"][0]
    while int(t.get_value()) < opts["flags"].total_iterations:
        inputs, train_idx = train_sampler()
        train_cost = network["backprop"](*inputs)

        inputs, test_idx = test_sampler()
        test_cost = network["cost"](*inputs)

        print train_idx, test_idx, train_cost, test_cost[0]

    return network


def train_network1(opts, network, h5_data, train_vids, test_vids, run):
    """Train the network."""
    # setup the data samplers
    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, train_vids)
    train_sampler = sampler

    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, test_vids, len(train_vids))
    test_sampler = sampler
    opts["rng"] = np.random.RandomState(123)
    train_inputs, train_idx = train_sampler()
    test_inputs, test_idx = test_sampler()

    save_dict = None

    t = network["lr_update"]["params"][0]
    while int(t.get_value()) < opts["flags"].total_iterations:
        train_cost = network["backprop"](*train_inputs)
        test_cost = network["cost"](*test_inputs)
        weights = np.absolute(network["params"][3].get_value()).mean()

        print "%d: " % int(t.get_value()), train_idx, test_idx,\
            train_cost, test_cost[0], weights
        # print "%f, %f\n" % (train_cost, test_cost[0])
        if run == 0 and int(t.get_value()) == 500:
            rng_state, save_dict = _save_current_state(opts, network)

    return network, save_dict


def train_network2(opts, network, h5_data, train_vids, test_vids, run):
    """Train the network."""
    # setup the data samplers
    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, train_vids)
    train_sampler = sampler

    def sampler():
        return hantman_mouse.random_mini_batch(
            opts, h5_data, test_vids, len(train_vids))
    test_sampler = sampler

    save_dict = None
    train_inputs, train_idx = train_sampler()
    test_inputs, test_idx = test_sampler()

    t = network["lr_update"]["params"][0]
    while int(t.get_value()) < opts["flags"].total_iterations:

        train_cost = network["backprop"](*train_inputs)
        test_cost = network["cost"](*test_inputs)
        weights = np.absolute(network["params"][3].get_value()).mean()

        print "%d: " % int(t.get_value()), train_idx, test_idx,\
            train_cost, test_cost[0], weights
        # print "%f, %f\n" % (train_cost, test_cost[0])
        # if run == 0 and int(t.get_value()) == 40:
        #     opts["rng"].set_state(np.random.RandomState(123).get_state())

        #     rng_state, save_dict = _save_current_state(opts, network)

        if run == 0 and int(t.get_value()) == 50:
            # change the rng seed here. This should shake things up.
            # import pdb; pdb.set_trace()
            opts["rng"] = np.random.RandomState(123)
            network = hantman_multi_lstm.create_network(
                opts, num_input, num_classes)

            network["lr_update"]["params"][0].set_value(50)
            # rng_state, save_dict = _save_current_state(opts, network)
            # network = _load_previous_state(opts, network, save_dict)
            # t = network["lr_update"]["params"][0]
            t = network["lr_update"]["params"][0]

            # opts["rng"].set_state(np.random.RandomState(123).get_state())
            # run = 1

    return network, save_dict


def _save_current_state(opts, network):
    rng_state = opts["rng"].get_state()
    params_dict = dict()
    for param in network["params"]:
        params_dict[param.name] = param.get_value()
    grads_dict = dict()
    for grad in network["grad_vals"]:
        grads_dict[grad.name] = grad.get_value()

    optim_dict = dict()
    for optim in network["optim_state"]:
        optim_dict[optim.name] = optim.get_value()

    save_dict = {
        "rng": rng_state,
        "params_dict": params_dict,
        "grads_dict": grads_dict,
        "optim_dict": optim_dict,
        "t": network["lr_update"]["params"][0].get_value(),
    }

    return rng_state, save_dict


def _load_previous_state(opts, network, save_dict):
    opts["rng"].set_state(save_dict["rng"])

    params_dict = save_dict["params_dict"]
    for param in network["params"]:
        param.set_value(params_dict[param.name])

    grads_dict = save_dict["grads_dict"]
    for grad in network["grad_vals"]:
        grad.set_value(grads_dict[grad.name])

    optim_dict = save_dict["optim_dict"]
    for optim in network["optim_state"]:
        optim.set_value(optim_dict[optim.name])

    network["lr_update"]["params"][0].set_value(save_dict["t"])
    return network


if __name__ == "__main__":
    print sys.argv
    # opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
    FLAGS = gflags.FLAGS

    opts = dict()
    opts["eps"] = np.finfo(np.float32).eps
    opts["argv"] = sys.argv
    # parse the inputs
    FLAGS(sys.argv)
    if arg_parsing.check_help(FLAGS) is True:
        sys.exit()

    if FLAGS.debug is True:
        opts["rng"] = np.random.RandomState(123)
    else:
        opts["rng"] = np.random.RandomState(123)

    # create the output directory
    paths.setup_output_space(FLAGS.out_dir1)
    paths.setup_output_space(FLAGS.out_dir2)
    # paths.save_command(FLAGS.out_dir1)
    # paths.save_command(FLAGS.out_dir2)
    # joblib.dump(opts, os.path.join(FLAGS.out_dir1, "opts", "opts.npy"))
    # joblib.dump(opts, os.path.join(FLAGS.out_dir2, "opts", "opts.npy"))

    opts["flags"] = FLAGS

    # load the data
    with h5py.File(opts["flags"].filename, "r") as h5_data:
        exp_list = h5_data["experiments"].value

        num_input = h5_data["exps"][exp_list[0]]["reduced"].shape[2]
        num_classes = h5_data["exps"][exp_list[0]]["labels"].shape[2]
        network = hantman_multi_lstm.create_network(
            opts, num_input, num_classes)

        if opts["flags"].load_network is not None:
            print "Loading cached network: %s" % opts["flags"].load_network
            hantman_multi_lstm.load_network(
                opts, opts["flags"].load_network, network)

            save_dict = joblib.load(opts["flags"].load_network)
            opts["rng"].set_state(save_dict["rng"])
            train_vids = save_dict["train_vids"]
            test_vids = save_dict["test_vids"]

        else:
            exp_mask = hantman_mouse.mask_long_vids(h5_data, exp_list)

            # get 15 videos for debugging
            if opts["flags"].debug is True:
                debug_mask = np.zeros((exp_list.shape[0],), dtype="bool")
                debug_mask[:5] = True
                debug_mask[200:206] = True
                debug_mask[500:511] = True
                exp_mask = debug_mask * exp_mask

            # setup the train/test data
            train_vids, test_vids = hantman_mouse.setup_train_test_samples(
                opts, h5_data, exp_mask)

            # convert the idx to key ids
            train_vids = exp_list[exp_mask][train_vids]
            test_vids = exp_list[exp_mask][test_vids]

        # # save state before training
        # rng_state, params_dict, grads_dict, optim_dict =\
        #     _save_current_state(opts, network)

        # train the first network
        network, save_dict = train_network2(
            opts, network, h5_data, train_vids, test_vids, 0)

        ######################################################################
        # train the network again...
        # rng_state = opts["rng"].get_state()
        # train the second network, from a save point
        # opts["rng"].set_state(rng_state)
        print "Working on second network"
        opts["rng"].set_state(np.random.RandomState(123).get_state())
        network = hantman_multi_lstm.create_network(
            opts, num_input, num_classes)
        network["lr_update"]["params"][0].set_value(50)
        # network = _load_previous_state(
        #     opts, network, save_dict)

        # network["lr_update"]["params"][0].set_value(-1)

        network = train_network2(
            opts, network, h5_data, train_vids, test_vids, 1)
