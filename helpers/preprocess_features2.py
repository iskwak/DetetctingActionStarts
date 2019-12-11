"""Use the hdf5 data to create pre-processed features."""
import argparse
# import pickle
# import cPickle as pickle
import numpy as np
from scipy import signal
import helpers.paths as paths
import helpers.git_helper as git_helper
import os
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from sklearn.externals import joblib

rng = np.random.RandomState(123)

g_frame_offsets = [-20, -10, -5, 0, 5, 10, 20]
g_delta_offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
g_smooth_window = 10
g_smooth_std = 5
g_exp_dir = "/localhome/kwaki/data/hantman_pruned/"
g_use_one = False


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts["filename"] = ""
    opts["out_dir"] = ""
    opts["frame_offsets"] = g_frame_offsets
    opts["delta_offsets"] = g_delta_offsets
    opts["smooth_window"] = g_smooth_window
    opts["smooth_std"] = g_smooth_std
    opts["exp_dir"] = g_exp_dir
    opts["use_one"] = g_use_one
    return opts


def setup_opts(opts):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    parser = argparse.ArgumentParser(description="Parse and convert Hantman"
                                     " lab mouse mat files into a more python "
                                     "friendly structure")
    parser.add_argument("-f", "--filename", type=str, required=True,
                        help="matlab file to parse")
    # parser.add_argument("-o", "--outname", required=True, type=str,
    #                     help="output picke file name")
    parser.add_argument("-o", "--out_dir", required=True, type=str,
                        help="output directory for picke""d data")
    parser.add_argument("-e", "--exp_dir", type=str,
                        help="location of mat files")
    parser.add_argument("-O", "--frame-offsets", type=int, nargs="+",
                        default=g_frame_offsets,
                        help="Frame window offsets for the features")
    parser.add_argument("-d", "--deltas", type=int, nargs="+",
                        default=g_delta_offsets,
                        help="Delta frame window offsets")
    parser.add_argument("-w", "--window", type=int, default=g_smooth_window,
                        help="Size of the window for smoothing")
    parser.add_argument("-s", "--std", type=float, default=g_smooth_std,
                        help="Standard dev for smoothing")

    parser.add_argument("--use_one", dest="use_one", action="store_true")
    parser.set_defaults(use_one=g_use_one)

    # parser.add_argument("--one", dest="one", action="store_true")
    # parser.add_argument
    # parser.add_argument("
    command_args = parser.parse_args()

    opts["filename"] = command_args.filename
    opts["out_dir"] = command_args.out_dir
    opts["exp_dir"] = command_args.exp_dir
    opts["frame_offsets"] = command_args.frame_offsets
    opts["delta_offsets"] = command_args.deltas
    opts["smooth_window"] = command_args.window
    opts["smooth_std"] = command_args.std
    opts["use_one"] = command_args.use_one

    return opts


def concat_features(opts, features, start_idx, end_idx, frame_offsets,
                    delta_offsets):
    """Concatenate features."""
    num_steps = len(frame_offsets)

    all_features = []
    for i in range(num_steps):
        first_idx = start_idx + frame_offsets[i]
        last_idx = end_idx + frame_offsets[i]
        all_features.append(features[first_idx:last_idx])
        # expected = all_features[0].shape[0]
        # for j in range(len(all_features)):
        #     if all_features[j].shape[0] != expected:

    num_steps = len(delta_offsets)
    base_features = features[start_idx:end_idx]
    for i in range(num_steps):
        first_idx = start_idx + delta_offsets[i]
        last_idx = end_idx + delta_offsets[i]
        # all_features.append(base_features - features[first_idx:last_idx])
        temp = base_features - features[first_idx:last_idx]
        all_features.append(temp)
        # print "temp: %d" % temp.shape[0]
        # print "all_features: %d" % all_features[-1].shape[0]

    # concatenate all the features together to make the desired matrix
    concat_feat = np.concatenate(all_features, axis=1)
    return concat_feat


def update_running_stat(n, mean, var, data):
    """Given a data array, update the running mean/std."""
    # from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        var += delta * (x - mean)
    return n, mean, var


def collect_write_label_stats(file, all_exps, all_labels):
    """Collect and write label stats."""
    # for each label, write some stats
    label_names = ["Lift", "Hand", "Grab", "Sup", "Mouth", "Chew"]
    file.write(("label name,total,missed,mean number,mean frame,min frame,"
                "max frame,std frame,10 percentile,90 percentile\n"))
    # write the header
    for i in range(len(label_names)):
        # collect some stats on the labels
        file.write("%s" % label_names[i])
        behavior_frames = []
        num_behav_per_video = []
        moo = []
        for j in range(len(all_labels)):
            idxs = np.argwhere(all_labels[j][:, i])
            num_behav_per_video.append(idxs.size)
            if idxs.size > 0:
                behavior_frames += idxs[0].tolist()
                moo.append(idxs[0].tolist())
        behavior_frames = np.asarray(behavior_frames)
        # number of labelled behaviors
        file.write(",%d" % behavior_frames.size)
        # number of videos with no labels
        file.write(",%d" % (np.asarray(num_behav_per_video) == 0).sum())
        # mean number of labels per video
        file.write(",%f" % np.asarray(num_behav_per_video).mean())
        # mean frame behavior occurs
        file.write(",%f" % behavior_frames.mean())
        # min frame
        file.write(",%f" % behavior_frames.min())
        # max frame
        file.write(",%f" % behavior_frames.max())
        # std
        file.write(",%f" % behavior_frames.std())
        # 10 percentile
        file.write(",%f" % np.percentile(behavior_frames, 10))
        # 90 percentile
        file.write(",%f" % np.percentile(behavior_frames, 90))
        file.write("\n")

    return


def filter_video(opts, h5exp):
    """Filter videos."""
    # short videos (less than 1500 frames)
    labels = h5exp["labels"].value
    filter = ""
    if labels.shape[0] < 1500:
        filter += "short"
    # no labels, some videos have no labels
    label_idx = np.argwhere(labels)
    if label_idx.size == 0:
        if filter != "":
            filter += ","
        filter += "none"
        return filter
    # early behavior. First behavior shows up before frame 50
    if label_idx[0][0] < 50:
        if filter != "":
            filter += ","
        filter += "early"
    # if np.all(label_idx[:, 0] >= 1500):
    #     if filter != "":
    #         filter += ","
    #     filter += "late"

    # multiple chews.
    if (label_idx[:, 1] == 5).sum() > 1:
        if filter != "":
            filter += ","
        filter += "multiple"

    return filter


def smooth_data(opts, org_labels):
    """Apply gaussian smoothing to the labels."""
    smooth_window = opts['smooth_window']
    smooth_std = opts['smooth_std']

    if smooth_window == 0:
        return org_labels

    # org_labels = labels
    labels = np.zeros(org_labels.shape)
    # loop over the columns and convolve
    conv_filter = signal.gaussian(smooth_window, std=smooth_std)
    # print conv_filter.shape
    # print labels.shape
    for i in range(labels.shape[1]):
        labels[:, i] = np.convolve(org_labels[:, i], conv_filter, 'same')
        # labels[:, i] = org_labels[:, i]
    # scale the labels a bit
    # labels = labels * 0.9
    # labels = labels + 0.01
    return labels


def get_exp_data(ipca, exp_file):
    """Process the exp_file."""
    # crop the data
    org_labels = exp_file["labels"].value
    label_idx = np.argwhere(org_labels)

    # based of this point, pick a crop point.
    last_frame = label_idx[-1][0]

    # randomly select a number of frames after this last from for the end of
    # the video
    last_frame = last_frame + 10 + rng.randint(0, 30)
    org_labels = org_labels[:last_frame, :]

    org_features = exp_file["hoghof"].value
    org_features = org_features[:last_frame, :]

    reduced = org_features
    reduced = np.dot(org_features - ipca.mean_,
                     ipca.components_[:500, :].T)
    reduced = reduced.reshape((reduced.shape[0], 1, reduced.shape[1]))

    pos_features = exp_file["pos_features"].value
    pos_features = pos_features[:last_frame, :]

    # process the group name, to get the date and mouse number.
    filename = exp_file.filename
    exp_name = os.path.basename(filename).encode("ascii", "ignore")
    split_name = exp_name.split("_")
    if len(split_name) != 3:
        print("ERROR EXPERIMENT NAME")
        import pdb; pdb.set_trace()
    mouse = split_name[0]  # .encode("ascii", "ignore")
    date = split_name[1]  # .encode("ascii", "ignore")

    labels = smooth_data(opts, org_labels)
    # labels = org_labels
    labels = labels.reshape((labels.shape[0], 1, labels.shape[1]))

    return org_features, reduced, pos_features, exp_name, mouse, date, labels,\
        org_labels


def preprocess_features(opts, log, skip_log, out_data, exp_dir):
    """Preprocess the features (compute and apply means and so on.)."""
    # get files
    base_exp_dir = os.path.join(opts["filename"], "exps")
    exps = os.listdir(base_exp_dir)
    exps.sort()
    num_exps = len(exps)

    # add the features group
    # out_data.create_group("features")
    # out_data.create_group("labels")
    out_data.create_group("exps")

    # ipca = IncrementalPCA(n_components=1500, batch_size=1500)
    ipca = joblib.load(("/media/drive1/data/hantman_processed/"
                        "relative_19window2/ipca/ipca2.npy"))
    # all_feat = []
    # all_labels = []
    all_exps = []
    all_mice = []
    all_dates = []
    tic = time.time()
    for i in range(num_exps):
        # if i > 10:
        #     break
        exp = exps[i]
        if opts["use_one"] is True and "M173" not in exp:
            continue

        # first load the data
        with h5py.File(os.path.join(base_exp_dir, exp), "r") as exp_file:
            print("(%d, %d): %s" % (i, num_exps, exp))
            filter = filter_video(opts, exp_file)

            if filter != "":
                # print "(%d, %d): %s" % (i, num_exps, exp)
                # print filter
                # print "%s,%s" % (exp, filter)
                skip_log.write("%s,%s" % (exp, filter))
                continue
            # else use the experiment file
            org_features, reduced, pos_features, exp_name, mouse, date,\
                labels, org_labels = get_exp_data(ipca, exp_file)
            # get_exp_data(ipca, exp_file)

            if np.argwhere(labels).size == 0:
                import pdb; pdb.set_trace()

            # for scalability... lets hope we don't hit the limit of files in
            # a directory
            with h5py.File(os.path.join(exp_dir, exp), "w") as out_exp:
                out_exp["hoghof"] = org_features.astype("float32")
                out_exp["pos_features"] = pos_features.astype("float32")
                out_exp["hoghof_reduced"] = reduced.astype("float32")

                temp = reduced.reshape((reduced.shape[0], reduced.shape[2]))
                temp = np.concatenate([temp, pos_features], axis=1)
                temp = temp.reshape((temp.shape[0], 1, temp.shape[1]))
                out_exp["reduced"] = temp.astype("float32")

                # out_exp["labels"] = labels.astype("float32")
                out_exp["labels"] = labels.astype("float32")
                out_exp["org_labels"] = org_labels.astype("float32")
                out_exp["data"] = date
                out_exp["mouse"] = mouse
            out_data["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/")

            # out_data["features"][exp] = reduced
            # out_data["labels"][exp] = labels

            # all_feat.append(reduced.astype("float32"))
            # all_labels.append(labels.astype("float32"))
            all_exps.append(exp_name)
            all_mice.append(mouse)
            all_dates.append(date)

            print((time.time() - tic))
            tic = time.time()

    # store all the features in a big array, friendly for being used by the
    # lstm.
    # all_feat = np.concatenate(all_feat, axis=1)
    # all_feat = all_feat.astype("float32")
    # all_labels = np.concatenate(all_labels, axis=1)
    # all_labels = all_labels.astype("float32")
    # write the data to the hdf5 file
    # import pdb; pdb.set_trace()
    # out_data["features"] = all_feat
    # out_data["labels"] = all_labels
    out_data["experiments"] = all_exps
    out_data["mice"] = all_mice
    out_data["date"] = all_dates

    # import pdb; pdb.set_trace()
    return

if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    # create the output directory
    # paths.setup_output_space(opts["out_dir"])
    paths.create_dir(opts["out_dir"])
    paths.save_command(opts["out_dir"])

    # log the git information
    git_helper.log_git_status(
        os.path.join(opts["out_dir"], "00_git_status.txt"))

    exp_dir = os.path.join(opts["out_dir"], "exps")
    paths.create_dir(exp_dir)

    outname = os.path.join(opts["out_dir"], "data.hdf5")
    logname = os.path.join(opts["out_dir"], "00_log.txt")
    skipname = os.path.join(opts["out_dir"], "00_skipped.txt")
    with open(logname, "w") as log:
        with open(skipname, "w") as skip_log:
            with h5py.File(outname, "w") as out_data:
                preprocess_features(opts, log, skip_log, out_data, exp_dir)
