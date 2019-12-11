"""Helper script to collect stats on hantman data."""
import numpy
import argparse
import h5py
import scipy.io as sio
import helpers.paths as paths
# import helpers.git_helper as git_helper
import os
import time
import cv2

rng = numpy.random.RandomState(123)

# g_frame_offsets = [-20, -10, -5, 0, 5, 10, 20]
# g_delta_offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# g_smooth_window = 10
# g_smooth_std = 5
g_exp_dir = "/localhome/kwaki/data/hantman_pruned/"
g_all_exp_dir = "/mnt/"


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
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output directory for picke""d data")
    parser.add_argument("-e", "--exp_dir", type=str, required=True,
                        help="location of mat files")
    parser.add_argument("-a", "--all_exp", type=str, required=True,
                        help="location of hantman data in original format")
    # parser.add_argument
    # parser.add_argument("
    command_args = parser.parse_args()

    opts["filename"] = command_args.filename
    opts["out_dir"] = command_args.out_dir
    opts["exp_dir"] = command_args.exp_dir
    opts["all_exp"] = command_args.all_exp
    return opts


def find_org_exp(opts, h5exp, exp_name):
    """Find the original experiment location... This is slow..."""
    exps = list(h5exp.keys())
    if exp_name in exps:
        return h5exp[exp_name]["path"].value

    exp_path = paths.find_exp_dir(exp_name, opts["all_exp"])

    cur_exp = h5exp.create_group(exp_name)
    cur_exp["path"] = exp_path

    return exp_path


def get_labels(num_frames, rawmat):
    """Get the labels out of the rawmat."""
    label_names = ["Lift_labl_t0sPos", "Handopen_labl_t0sPos",
                   "Grab_labl_t0sPos", "Sup_labl_t0sPos",
                   "Atmouth_labl_t0sPos", "Chew_labl_t0sPos"]

    # convert the label format into a vector
    labels = numpy.zeros((num_frames, len(label_names)), dtype=numpy.float32)
    for i in range(len(label_names)):
        t0s_str = label_names[i]
        t0s = rawmat[t0s_str]
        for k in range(t0s.size):
            labels[t0s[0][k], i] = 1

    return labels


def create_relative_features(trxdata, trx, cap):
    """Create relative position features."""
    x1 = numpy.asarray(trxdata["x1"])
    y1 = numpy.asarray(trxdata["y1"])
    x2 = numpy.asarray(trxdata["x2"])
    y2 = numpy.asarray(trxdata["y2"])

    # for some reason there are 5 copies of the info in trx?
    # want to use, perch, food, mouth
    # ... don"t really understand nested structs in matfiles and python
    # these are associated with x1,y1
    food = trx["trx"][0][0]["arena"]["food"][0][0][0]
    mouth = trx["trx"][0][0]["arena"]["mouth"][0][0][0]
    perch = trx["trx"][0][0]["arena"]["perch"][0][0][0]
    # these are associated with x2,y2
    foodfront = trx["trx"][0][0]["arena"]["foodfront"][0][0][0]
    mouthfront = trx["trx"][0][0]["arena"]["mouthfront"][0][0][0]
    # no perchfront

    # the arena positions are not relative to the concatenated
    # features
    import pdb; pdb.set_trace()
    width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    foodfront[0] = foodfront[0] + width / 2
    mouthfront[0] = mouthfront[0] + width / 2
    # foodfront[0] = foodfront[0] + width // 2
    # mouthfront[0] = mouthfront[0] + width // 2

    # next create the relative position features
    num_feat = x1.shape[0]
    features = numpy.zeros((num_feat, 10), dtype=numpy.float32)
    for i in range(num_feat):
        # first frame
        features[i, 0] = x1[i] - perch[0]
        features[i, 1] = y1[i] - perch[1]
        features[i, 2] = x1[i] - mouth[0]
        features[i, 3] = y1[i] - mouth[1]
        features[i, 4] = x1[i] - food[0]
        features[i, 5] = y1[i] - food[1]
        # second frame
        features[i, 6] = x2[i] - mouthfront[0]
        features[i, 7] = y2[i] - mouthfront[1]
        features[i, 8] = x2[i] - foodfront[0]
        features[i, 9] = y2[i] - foodfront[1]

    # pos_features = numpy.concatenate([x1, y1, x2, y2, features], axis=1)
    pos_features = features

    return pos_features, perch, mouth, food, mouthfront, foodfront


def process_mat(opts, logfile, exp_name, h5exp, rawmat, trxdata):
    """Process the matfile."""
    # import pdb; pdb.set_trace()
    print("\tSearching for the experiment...")
    tic = time.time()
    org_exp = find_org_exp(opts, h5exp, exp_name)
    print("\t%s" % org_exp)
    print("\tTook %f seconds" % (time.time() - tic))

    print("\tLoading trx.mat...")
    tic = time.time()
    trxmat = sio.loadmat(os.path.join(org_exp, "trx.mat"))
    print("\tTook %f seconds" % (time.time() - tic))

    # print "\tLoading features.mat..."
    # tic = time.time()
    matname = os.path.join(org_exp, "features.mat")
    hoghof = sio.loadmat(matname)
    hoghof = hoghof["curFeatures"]
    # print "\tTook %f seconds" % (time.time() - tic)

    # print "\tLoading features.mat..."
    # tic = time.time()
    trxmat = sio.loadmat(os.path.join(org_exp, "trx.mat"))
    # print "\tTook %f seconds" % (time.time() - tic)

    num_frames = trxdata["x1"].size
    logfile.write(",%d" % num_frames)
    labels = get_labels(num_frames, rawmat)

    import pdb; pdb.set_trace()
    cap = cv2.VideoCapture(os.path.join(org_exp, "movie_comb.avi"))
    pos_features, perch, mouth, food, mouthfront, foodfront =\
        create_relative_features(trxdata, trxmat, cap)

    return org_exp, pos_features, hoghof, labels, num_frames


def parse_hantman_mat(opts, h5file, matfile, logfile):
    """Parse hantman matifle."""
    # each index in this mat file is an experiment (video). For each video,
    # create feature and label matrix. For label information, check the
    # rawdata field. For the trajectory data, check trxdata.
    num_experiments = matfile["trxdata"].size
    exp_names = [exp[0]["exp"][0] for exp in matfile["rawdata"]]
    exp_idx = numpy.argsort(exp_names)

    # get the h5keys
    if "exp" not in list(h5file.keys()):
        h5exp = h5file.create_group("exp")
    else:
        h5exp = h5file["exp"]
    # h5keys = h5file["exp"].keys()

    # keep track of skipped files
    # early_lift_exps = []
    # no_label_exps = []

    # sub_exp = range(num_experiments)
    count = 0
    for i in exp_idx:
        print("%d of %d" % (count, num_experiments))
        print("%d" % i)
        exp_name = matfile["rawdata"][i][0]["exp"][0]
        print("\t%s" % exp_name)
        logfile.write("%s" % exp_name)

        # if "M134" in exp_name or "M173" in exp_name or "M174" in exp_name:
        #     continue

        tic = time.time()
        org_exp, pos_features, hoghof, labels, num_frames =\
            process_mat(opts, logfile, exp_name,
                        h5exp,
                        matfile["rawdata"][i],
                        matfile["trxdata"][0][i])

        # load the JAABA classifier scores
        # scores, postproc = process_scores(opts, logfile, org_exp, num_frames)

        # write the data to h5 files (one for each experiment)
        out_file = os.path.join(opts["out_dir"], "exps", exp_name)
        with h5py.File(out_file, "w") as exp_file:
            # group = exp_file.create_group(exp_name)
            # group["pos_features"] = pos_features
            exp_file["pos_features"] = pos_features
            exp_file["hoghof"] = hoghof
            exp_file["labels"] = labels
            # exp_file["scores"] = scores
            # exp_file["post_processed"] = postproc
            trialtype = matfile["rawdata"][i][0]["trialtype"][0]
            trialtype = trialtype.encode("ascii", "ignore")
            exp_file.attrs["trail_type"] = trialtype
        print("Processing took: %f seconds" % (time.time() - tic))
        logfile.write("\n")
        logfile.flush()
        count = count + 1
    # END for i in range(num_experiments):

    return


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts["filename"] = ""
    opts["out_dir"] = ""
    opts["exp_dir"] = g_exp_dir
    opts["all_exp"] = g_all_exp_dir
    return opts

if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    # create the output directory
    paths.create_dir(opts["out_dir"])
    paths.create_dir(os.path.join(opts["out_dir"], "exps"))
    paths.save_command(opts["out_dir"])

    # log the git information
    # git_helper.log_git_status(
    #     os.path.join(opts["out_dir"], "00_git_status.txt"))

    # try to load the locations of the original experiments.
    h5filename = os.path.join(opts["out_dir"], "00_exp_cache.hdf5")
    h5file = h5py.File(h5filename, "a")

    # load the mat file
    matfile = sio.loadmat(opts["filename"])

    logfilename = os.path.join(opts["out_dir"], "00_log.txt")
    with open(logfilename, "w") as log:
        parse_hantman_mat(opts, h5file, matfile, log)

    h5file.close()
