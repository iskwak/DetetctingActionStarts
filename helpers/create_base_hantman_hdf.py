"""Use the hdf5 data to create pre-processed features."""
# import argparse
# import pickle
# import cPickle as pickle
import sys
import numpy
# from scipy import signal
import helpers.paths as paths
import helpers.git_helper as git_helper
import os
import h5py
import gflags
# from sklearn.decomposition import IncrementalPCA
# import shutil
import time
# from sklearn.externals import joblib
import helpers.arg_parsing as arg_parsing

gflags.DEFINE_string("input_dir", None, "Base directory with the experiments.")
gflags.DEFINE_string("out_dir", None, "Ouput directory path.")


def _setup_opts(argv):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)
    return opts


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
            idxs = numpy.argwhere(all_labels[j][:, i])
            num_behav_per_video.append(idxs.size)
            if idxs.size > 0:
                behavior_frames += idxs[0].tolist()
                moo.append(idxs[0].tolist())
        behavior_frames = numpy.asarray(behavior_frames)
        # number of labelled behaviors
        file.write(",%d" % behavior_frames.size)
        # number of videos with no labels
        file.write(",%d" % (numpy.asarray(num_behav_per_video) == 0).sum())
        # mean number of labels per video
        file.write(",%f" % numpy.asarray(num_behav_per_video).mean())
        # mean frame behavior occurs
        file.write(",%f" % behavior_frames.mean())
        # min frame
        file.write(",%f" % behavior_frames.min())
        # max frame
        file.write(",%f" % behavior_frames.max())
        # std
        file.write(",%f" % behavior_frames.std())
        # 10 percentile
        file.write(",%f" % numpy.percentile(behavior_frames, 10))
        # 90 percentile
        file.write(",%f" % numpy.percentile(behavior_frames, 90))
        file.write("\n")

    return


def filter_video(opts, h5exp):
    """Filter videos."""
    # short videos (less than 1500 frames)
    labels = h5exp["labels"].value
    filter = ""
    # if labels.shape[0] < 1000:
    #     filter += "short"
    # no labels, some videos have no labels
    label_idx = numpy.argwhere(labels)
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
    if numpy.all(label_idx[:, 0] >= 1500):
        if filter != "":
            filter += ","
        filter += "late"

    # multiple chews.
    if (label_idx[:, 1] == 5).sum() > 1:
        if filter != "":
            filter += ","
        filter += "multiple"

    return filter


def get_exp_data(opts, exp_file):
    """Process the exp_file."""
    # crop the data
    labels = exp_file["labels"].value
    label_idx = numpy.argwhere(labels)

    # based of this point, pick a crop point.
    last_frame = label_idx[-1][0]

    # randomly select a number of frames after this last from for the end of
    # the video
    last_frame = last_frame + 50 + opts["rng"].randint(0, 50)
    labels = labels[:last_frame, :]

    # change of plans, only take the position features. Everything else will
    # and can be easily regenerated.
    positions = exp_file["pos_features"]
    positions = positions[:last_frame, :]
    # no longer reshaping... To do multi gpu, it is easier if the data is
    # ordered differently. So for now, just make it a later functions job
    # to reshape the data.

    # process the group name, to get the date and mouse number.
    filename = exp_file.filename
    exp_name = os.path.basename(filename).encode("ascii", "ignore")
    split_name = exp_name.split("_")
    if len(split_name) != 3:
        print("ERROR EXPERIMENT NAME")
        import pdb; pdb.set_trace()
    mouse = split_name[0]  # .encode("ascii", "ignore")
    date = split_name[1]  # .encode("ascii", "ignore")

    video_name = os.path.join(exp_name, "movie_comb.avi")
    return positions, exp_name, video_name, mouse, date, labels


def preprocess_features(opts, log, skip_log, out_data, exp_dir):
    """Preprocess the features (compute and apply means and so on.)."""
    # get files
    base_exp_dir = os.path.join(opts["flags"].input_dir, "exps")
    exps = os.listdir(base_exp_dir)
    exps.sort()
    num_exps = len(exps)
    print("Total experiments: %d" % num_exps)
    # add the features group
    # out_data.create_group("features")
    # out_data.create_group("labels")
    out_data.create_group("exps")

    # all_feat = []
    # all_labels = []
    all_exps = []
    all_mice = []
    all_dates = []
    # tic = time.time()
    for i in range(num_exps):
        # if i > 10:
        #     break
        exp = exps[i]

        # first load the data
        with h5py.File(os.path.join(base_exp_dir, exp), "r") as exp_file:
            # print("(%d, %d): %s" % (i, num_exps, exp))

            # if the mouse is M135, skip it.
            if "M135" in exp:
                filter = "M135"
            else:
                # if not M135, apply the filtering.
                filter = filter_video(opts, exp_file)

            if filter != "":
                # print "(%d, %d): %s" % (i, num_exps, exp)
                # print filter
                # print "%s,%s" % (exp, filter)
                skip_log.write("%s,%s\n" % (exp, filter))
                continue

            # else use the experiment file
            positions, exp_name, video_name, mouse, date, labels =\
                get_exp_data(opts, exp_file)

            if numpy.argwhere(labels).size == 0:
                import pdb; pdb.set_trace()

            # for scalability... lets hope we don't hit the limit of files in
            # a directory
            with h5py.File(os.path.join(exp_dir, exp), "w") as out_exp:
                out_exp["positions"] = positions
                out_exp["labels"] = labels.astype("float32")
                out_exp["date"] = date
                out_exp["mouse"] = mouse
                out_exp["video_name"] = video_name

            out_data["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/")

            all_exps.append(exp_name)
            all_mice.append(mouse)
            all_dates.append(date)

            # print((time.time() - tic))
            # tic = time.time()

    out_data["exp_names"] = all_exps
    out_data["mice"] = all_mice
    out_data["date"] = all_dates

    # import pdb; pdb.set_trace()
    return


def main(opts):
    # create the output directory
    paths.create_dir(opts["flags"].out_dir)
    paths.save_command2(opts["flags"].out_dir, opts["argv"])

    # log the git information
    git_helper.log_git_status(
        os.path.join(opts["flags"].out_dir, "00_git_status.txt"))

    exp_dir = os.path.join(opts["flags"].out_dir, "exps")
    paths.create_dir(exp_dir)

    outname = os.path.join(opts["flags"].out_dir, "data.hdf5")
    logname = os.path.join(opts["flags"].out_dir, "00_log.txt")
    skipname = os.path.join(opts["flags"].out_dir, "00_skipped.txt")
    with open(logname, "w") as log:
        with open(skipname, "w") as skip_log:
            with h5py.File(outname, "w") as out_data:
                preprocess_features(opts, log, skip_log, out_data, exp_dir)


if __name__ == "__main__":
    opts = _setup_opts(sys.argv)
    main(opts)
