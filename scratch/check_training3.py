"""Mouse behavior spike classification."""
import gflags
import helpers.paths as paths
import os
import h5py
import sys
import helpers.arg_parsing as arg_parsing
import numpy as np
from sklearn.externals import joblib
import helpers.git_helper as git_helper
import helpers.hantman_mouse as hantman_mouse
import helpers.sequences_helper as sequences_helper
import models.hantman_hungarian as hantman_hungarian
# import time
import theano
# import helpers.DataLoader as DataLoader
# import copy
import helpers.post_processing as post_processing
import shutil

gflags.DEFINE_string("out_dir", None, "Directory of the training results.")
gflags.DEFINE_string("process_dir", None, "Directory to store processing"
                     " information.")
gflags.DEFINE_string("filename", None, "Feature data filename (hdf5).")
gflags.DEFINE_string("image_dir", None, "Directory for images to symlink.")
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
gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("filename")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(hantman_hungarian)


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
        # import pdb; pdb.set_trace()
        if temp_idx not in h5_data["exps"].keys():
            import pdb; pdb.set_trace()
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

        # outputs = network["cost"](*inputs)
        predict = network["predict_batch"](*inputs)
        predictions = predict

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


def compute_tpfp(label_dicts):
    """Compute the precision recall information."""
    f1_scores = []
    mean_f = 0
    for i in range(len(label_dicts)):
        tp = float(len(label_dicts[i]['dists']))
        fp = float(label_dicts[i]['fp'])
        fn = float(label_dicts[i]['fn'])
        precision = tp / (tp + fp + 0.0001)
        recall = tp / (tp + fn + 0.0001)
        f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
        print "label: %s" % label_dicts[i]['label']
        print "\tprecision: %f" % precision
        print "\trecall: %f" % recall
        print "\tfscore: %f" % f1_score
        mean_f += f1_score
        f1_scores.append(f1_score)
    mean_f = (mean_f / len(label_dicts))
    print "mean score: %f" % mean_f

    return mean_f, f1_scores


def check_network(opts, network, h5_data, train_vids, test_vids):
    """Check the network."""
    # make sequence predictions
    base_dir = opts["flags"].out_dir
    network_dirs = os.listdir(os.path.join(base_dir, "networks"))
    network_iters = [int(val) for val in network_dirs]
    network_iters.sort()

    # opts["flags"].out_dir = "/home/ikwak/research/moo"
    opts["flags"].out_dir = opts["flags"].process_dir
    paths.setup_output_space(opts["flags"].out_dir)
    plot_dir = os.path.join(opts["flags"].out_dir, "plots")
    paths.create_dir(plot_dir)
    # copy over some extra templates
    shutil.copy("templates/require.js", plot_dir)
    shutil.copy("templates/loss_fscore.html", plot_dir)
    shutil.copy("templates/loss_fscore.js", plot_dir)
    print "copying frames/templates..."
    sequences_helper.copy_main_graphs(opts)

    base_train = os.path.join(opts["flags"].out_dir, "predictions", "train")
    # train_experiments = exp_list[train_vids]
    train_experiments = train_vids
    sequences_helper.copy_experiment_graphs(
        opts, base_train, train_experiments)

    base_test = os.path.join(opts["flags"].out_dir, "predictions", "test")
    # test_experiments = exp_list[train_vids]
    test_experiments = test_vids
    sequences_helper.copy_experiment_graphs(
        opts, base_test, test_experiments)

    print "done"

    loss_filename = os.path.join(plot_dir, "loss.csv")
    with open(loss_filename, "w") as outfile:
        outfile.write("iteration,training loss,test loss,"
                      "train fscore,test fscore,"
                      "train lift,train hand,train grab,train sup,"
                      "train mouth,train chew,"
                      "test lift,test hand,test grab,test sup,"
                      "test mouth,test chew\n")

    for i in network_iters:
        opts["flags"].load_network = os.path.join(
            base_dir, "networks", "%d" % i, "network.npy")
        print opts["flags"].load_network

        save_dict = joblib.load(opts["flags"].load_network)
        hantman_hungarian.load_network(
            opts, opts["flags"].load_network, network)
        opts["rng"].set_state(save_dict["rng"])

        if "train_cost" not in save_dict.keys():
            train_cost = 0
            test_cost = 0
        else:
            train_cost = save_dict["train_cost"]
            test_cost = save_dict["test_cost"][0]

        _process_full_sequences(
            opts, network, h5_data, train_vids, test_vids)

        label_dicts = post_processing.process_outputs(
            base_train, "")
        train_f, train_scores = compute_tpfp(label_dicts)

        label_dicts = post_processing.process_outputs(
            base_test, "")
        test_f, test_scores = compute_tpfp(label_dicts)

        with open(loss_filename, "a") as outfile:
            outfile.write("%d,%f,%f,%f,%f" %
                          (i, train_cost, test_cost, train_f, test_f))
            for score in train_scores:
                outfile.write(",%f" % score)
            for score in test_scores:
                outfile.write(",%f" % score)
            outfile.write("\n")

    return


def initialize_network_helper(opts, h5_data, exp_list):
    """Setup the network."""
    num_input = h5_data["exps"][exp_list[0]]["reduced"].shape[2]
    num_classes = h5_data["exps"][exp_list[0]]["labels"].shape[2]
    network = hantman_hungarian.create_network(
        opts, num_input, num_classes)

    if opts["flags"].load_network is not None:
        print "Loading cached network: %s" % opts["flags"].load_network
        hantman_hungarian.load_network(
            opts, opts["flags"].load_network, network)

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

    opts["flags"] = FLAGS

    opts["flags"].load_network = os.path.join(
        opts["flags"].out_dir, "networks", "0", "network.npy")

    if opts["flags"].load_network is not None:
        save_dict = joblib.load(opts["flags"].load_network)
        opts["rng"].set_state(save_dict["rng"])

    # load the data
    with h5py.File(opts["flags"].filename, "r") as h5_data:
        exp_list = h5_data["experiments"].value

        # log the git information
        git_helper.log_git_status(
            os.path.join(opts["flags"].out_dir, "git_status.txt"))

        exp_mask = hantman_mouse.mask_long_vids(h5_data, exp_list)
        exp_mask = [True for i in exp_mask]
        import pdb; pdb.set_trace()

        # # get 15 videos for debugging
        # if opts["flags"].debug is True:
        #     debug_mask = np.zeros((exp_list.shape[0],), dtype="bool")
        #     debug_mask[:5] = True
        #     debug_mask[200:206] = True
        #     debug_mask[500:511] = True
        #     exp_mask = debug_mask * exp_mask

        # load the network here
        network = initialize_network_helper(opts, h5_data, exp_list)

        # pre-process data
        print opts["flags"].load_network
        if opts["flags"].load_network is None:
            train_vids, test_vids = hantman_mouse.setup_train_test_samples(
                opts, h5_data, exp_mask)
            # convert the idx to key ids
            train_vids = exp_list[exp_mask][train_vids]
            test_vids = exp_list[exp_mask][test_vids]
        else:
            print "loading cached train/test vids"
            save_dict = joblib.load(opts["flags"].load_network)
            train_vids = save_dict["train_vids"]
            test_vids = save_dict["test_vids"]

        check_network(opts, network, h5_data, train_vids, test_vids)
        print "hi"
