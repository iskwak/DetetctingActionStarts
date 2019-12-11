"""Convert matlab mouse data into a pickle."""
import argparse
# import pickle
# import cPickle as pickle
from sklearn.externals import joblib
import numpy
import scipy.io as sio
from scipy import signal
import cv2
import helpers.paths as paths
import helpers.git_helper as git_helper
import os
# import shutil

rng = numpy.random.RandomState(123)

g_frame_offsets = [-20, -10, -5, 0, 5, 10, 20]
g_delta_offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
g_smooth_window = 10
g_smooth_std = 5
g_exp_dir = '/localhome/kwaki/data/hantman_pruned/'


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts['filename'] = ''
    opts['out_dir'] = ''
    opts['frame_offsets'] = g_frame_offsets
    opts['delta_offsets'] = g_delta_offsets
    opts['smooth_window'] = g_smooth_window
    opts['smooth_std'] = g_smooth_std
    opts['exp_dir'] = g_exp_dir
    return opts


def setup_opts(opts):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    parser = argparse.ArgumentParser(description='Parse and convert Hantman'
                                     ' lab mouse mat files into a more python '
                                     'friendly structure')
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='matlab file to parse')
    # parser.add_argument('-o', '--outname', required=True, type=str,
    #                     help='output picke file name')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='output directory for picke''d data')
    parser.add_argument('-e', '--exp_dir', type=str,
                        help='location of mat files')
    parser.add_argument('-O', '--frame-offsets', type=int, nargs='+',
                        default=g_frame_offsets,
                        help='Frame window offsets for the features')
    parser.add_argument('-d', '--deltas', type=int, nargs='+',
                        default=g_delta_offsets,
                        help='Delta frame window offsets')
    parser.add_argument('-w', '--window', type=int, default=g_smooth_window,
                        help='Size of the window for smoothing')
    parser.add_argument('-s', '--std', type=float, default=g_smooth_std,
                        help='Standard dev for smoothing')
    # parser.add_argument
    # parser.add_argument("
    command_args = parser.parse_args()

    opts['filename'] = command_args.filename
    opts['out_dir'] = command_args.out_dir
    opts['exp_dir'] = command_args.exp_dir
    opts['frame_offsets'] = command_args.frame_offsets
    opts['delta_offsets'] = command_args.deltas
    opts['smooth_window'] = command_args.window
    opts['smooth_std'] = command_args.std
    return opts


def create_relative_features(trx, cap, x1, y1, x2, y2):
    """Create relative position features."""
    # for some reason there are 5 copies of the info in trx?
    # want to use, perch, food, mouth
    # ... don't really understand nested structs in matfiles and python
    # these are associated with x1,y1
    food = trx['trx'][0][0]['arena']['food'][0][0][0]
    mouth = trx['trx'][0][0]['arena']['mouth'][0][0][0]
    perch = trx['trx'][0][0]['arena']['perch'][0][0][0]
    # these are associated with x2,y2
    foodfront = trx['trx'][0][0]['arena']['foodfront'][0][0][0]
    mouthfront = trx['trx'][0][0]['arena']['mouthfront'][0][0][0]
    # no perchfront

    # the arena positions are not relative to the concatenated
    # features
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    foodfront[0] = foodfront[0] + width / 2
    mouthfront[0] = mouthfront[0] + width / 2
    print("CHECK DIVISION HERE")
    import pdb; pdb.set_trace()

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

    return features


def process_video(opts, label_names, rawdata, trxdata,
                  frame_offsets, delta_offsets):
    """Process a single video from the hantman dataset."""
    # returns exp name, featues and labels
    exp_name = rawdata['exp']
    num_frames = trxdata['x1'].size

    # get the range of the range of the frames (can be negative)
    min_idx, max_idx, crop_first, crop_last, early_lift = find_first_last(
        opts, rawdata, label_names, num_frames, frame_offsets, delta_offsets)

    # before cropping/padding, create a label matrix, of the same size as the
    # feature matrix... easier to chop things up later
    labels = numpy.zeros((num_frames, len(label_names)), dtype=numpy.float32)
    for j in range(len(label_names)):
        t0s_str = label_names[j]
        t0s = rawdata[t0s_str]
        for k in range(t0s.size):
            labels[t0s[0][k], j] = 1
    # org_labels = labels  # for debug purposes.
    # convert the trx data into a "feature" matrix
    # asarray, with the list seems to give a strange shape... just convert to
    # numpy arrays and concatenate
    x1 = numpy.asarray(trxdata['x1'])
    y1 = numpy.asarray(trxdata['y1'])
    x2 = numpy.asarray(trxdata['x2'])
    y2 = numpy.asarray(trxdata['y2'])
    org_features = numpy.concatenate([x1, y1, x2, y2], axis=1)
    pos_features = org_features

    # get the landmarks here
    # exp_dir = ''
    # exp_dir = '/media/drive1/data/hantman_pruned/' + exp_name[0]
    # exp_dir = '/localhome/kwaki/data/hantman_pruned/' + exp_name[0]
    exp_dir = opts['exp_dir'] + exp_name[0]
    # exp_dir = 'C:/Users/ikwak/Desktop/hantman/' + exp_name[0]
    cap = cv2.VideoCapture(exp_dir + '/movie_comb.avi')
    trx = sio.loadmat(exp_dir + '/trx.mat')
    pos_features = create_relative_features(trx, cap, x1, y1, x2, y2)

    # next create video features
    # vid_features = create_video_features(trx, cap, x1, y1, x2, y2)
    # cap.release()
    # features = numpy.concatenate((vid_features, pos_features), axis=1)

    # after creating the label matrix (which should be the same size as the
    # feature matrix), pad/crop them.
    pos_features, start_idx, end_idx = pad_array(opts, min_idx, max_idx,
                                                 crop_first, crop_last,
                                                 pos_features)
    # vid_features, start_idx, end_idx = pad_array(opts, min_idx, max_idx,
    #                                              crop_first, crop_last,
    #                                              vid_features)
    labels, start_idx, end_idx = pad_array(opts, min_idx, max_idx,
                                           crop_first, crop_last, labels)
    # next create the desired features
    concat_feat = concat_features(opts, pos_features, start_idx, end_idx,
                                  frame_offsets, delta_offsets)
    frame_idx = list(range(crop_first, crop_last))

    if len(frame_idx) != labels.shape[0]:
        import pdb; pdb.set_trace()
    # if end_idx > num_frames:
    #     frame_idx = frame_idx[start_idx:num_frames]
    #     labels = labels[start_idx:num_frames]
    # else:
    #     frame_idx = frame_idx[start_idx:end_idx]
    #     labels = labels[start_idx:end_idx]
    frame_idx = frame_idx[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    # vid_features = vid_features[start_idx:end_idx]
    # all_features = numpy.concatenate((vid_features, concat_feat), axis=1)
    all_features = concat_feat
    # import pdb; pdb.set_trace()

    # smooth out the labels?
    labels = smooth_data(opts, labels)
    crops = {'crops': [crop_first, crop_last], 'idx': [start_idx, end_idx]}

    return exp_name, exp_dir, labels, all_features, crops, frame_idx,\
        early_lift


def create_video_features(trx, cap, x1, y1, x2, y2):
    """Create image based features (just the frames)."""
    # for some reason there are 5 copies of the info in trx?
    # want to use, perch, food, mouth
    # ... don't really understand nested structs in matfiles and python
    # these are associated with x1,y1
    # food = trx['trx'][0][0]['arena']['food'][0][0][0]
    # mouth = trx['trx'][0][0]['arena']['mouth'][0][0][0]
    # perch = trx['trx'][0][0]['arena']['perch'][0][0][0]
    # # these are associated with x2,y2
    # foodfront = trx['trx'][0][0]['arena']['foodfront'][0][0][0]
    # mouthfront = trx['trx'][0][0]['arena']['mouthfront'][0][0][0]
    # # no perchfront

    # # the arena positions are not relative to the concatenated
    # # features
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # foodfront[0] = foodfront[0] + width/2
    # mouthfront[0] = mouthfront[0] + width/2

    # build crops around the hands, and might as well do it around the
    # landmarks?
    halfsize = 35
    feat_size = (halfsize * 2) * (halfsize * 2)
    num_feat = x1.shape[0]
    features = numpy.zeros((num_feat, feat_size * 2), dtype=numpy.float32)
    for i in range(num_feat):
        ret, frame = cap.read()
        # make the frame grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # make some crops
        left_x = x1[i] - halfsize
        right_x = x1[i] + halfsize
        bottom_y = y1[i] - halfsize
        top_y = y1[i] + halfsize
        crop1 = gray_frame[bottom_y:top_y, left_x:right_x]
        # cv2.imwrite('test1.png', crop1)

        left_x = x2[i] - halfsize
        right_x = x2[i] + halfsize
        bottom_y = y2[i] - halfsize
        top_y = y2[i] + halfsize
        crop2 = gray_frame[bottom_y:top_y, left_x:right_x]
        # cv2.imwrite('test2.png', crop2)
        # feat1 = crop1.reshape((feat_size,))
        # feat2 = crop2.reshape((feat_size,))
        # features[i,:] = numpy.concatenate((feat1, feat2))
        # hmmm... prob okay for features, but easier to reshape later
        cat_image = numpy.concatenate((crop1, crop2), axis=1)
        # cv2.imwrite("test.png", cat_image)
        features[i, :] = cat_image.reshape((feat_size * 2,)) / 255.0

    return features


def smooth_data(opts, org_labels):
    """Apply gaussian smoothing to the labels."""
    smooth_window = opts['smooth_window']
    smooth_std = opts['smooth_std']

    if smooth_window == 0:
        return org_labels

    # org_labels = labels
    labels = numpy.zeros(org_labels.shape)
    # loop over the columns and convolve
    conv_filter = signal.gaussian(smooth_window, std=smooth_std)
    # print conv_filter.shape
    # print labels.shape
    for i in range(labels.shape[1]):
        labels[:, i] = numpy.convolve(org_labels[:, i], conv_filter, 'same')
        # labels[:, i] = org_labels[:, i]
    # scale the labels a bit
    # labels = labels * 0.9
    # labels = labels + 0.01
    return labels


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
    concat_feat = numpy.concatenate(all_features, axis=1)
    return concat_feat


def pad_array(opts, min_idx, max_idx, crop_first, crop_last, data):
    """Given some bounds, pad or crop the data array."""
    # first check crop_last. easier to keep track of the sizes... in my head
    num_frames = data.shape[0]
    start_idx = min_idx - crop_first
    #   print min_idx - crop_first
    # end_idx = crop_last - crop_first - (crop_last - max_idx)
    # translating from original array bounds, into the cropped array
    # bounds. crop first is the new start of the array. so end_idx
    # needs to be relative to that
    end_idx = max_idx - crop_first
    # print "pad_array (start_idx, end_idx): (%d, %d)" % (start_idx, end_idx)
    if crop_last > num_frames:
        # if crop_last > num_frames, just pad the data array and nothing else
        # needs to change
        pad_amount = crop_last - num_frames
        data = numpy.lib.pad(data, ((0, pad_amount), (0, 0)), 'edge')
        print("\tpadding video")
    if crop_first < 0:
        # if crop_first is less than 0, pad data
        pad_amount = crop_first * -1
        data = numpy.lib.pad(data, ((pad_amount, 0), (0, 0)), 'edge')
        # adjust the bounds after padding
        crop_first = 0
        crop_last = crop_last + pad_amount
        # start_idx = start_idx + pad_amount
        # end_idx = end_idx + pad_amount

    # do the actual cropping
    data = data[crop_first:crop_last]
    return data, start_idx, end_idx


def find_first_last(opts, labels, label_names, num_frames,
                    frame_offsets, delta_offsets):
    """Helper function to find the first and last labeled frames."""
    # find the first and last labeled action frames.
    min_idx = num_frames
    max_idx = 0
    max_label = ""
    for j in range(len(label_names)):
        t0s_str = label_names[j]
        t0s = labels[t0s_str]
        # print t0s
        if t0s[0].size > 0 and t0s[0][0] < min_idx:
            min_idx = t0s[0][0]
        if t0s[0].size > 0 and t0s[0][-1] > max_idx:
            max_idx = t0s[0][-1]
            max_label = label_names[j]
    # if min_idx didn't change, then this means that there were no
    # behaviors.
    is_empty = False
    if min_idx == num_frames:
        min_idx = 0
        is_empty = True
    if max_idx == 0:
        max_idx = num_frames

    # if num_frames == 2501:
    #     import pdb; pdb.set_trace()
    print("\t%s" % max_label)
    # the first frame to start collecting information from is min_idx and the
    # last frame is max idx. However we don't want to just start at the first
    # behavior. Add some padding.
    # min_idx = min_idx - 30
    # just in case this data will be used for backwards passes, randomize
    # the padding
    max_idx = max_idx + 30 + rng.randint(30)

    # this finds the first and last labeled actions. The current default
    # behavior is to add a 30 frame pad to each side of the actions. Or we
    # should crop based on the desired window/delta offsets.
    # min_offset = min(window_offsets + delta_offsets)
    # crop_first = min([min_idx, min_idx + min_offset])
    # max_offset = max(window_offsets + delta_offsets)
    # print "max_idx: %d, crop_last: %d" % (max_idx, crop_last)

    # alternative cropping scheme.
    min_offset = min(frame_offsets + delta_offsets)
    max_offset = max(frame_offsets + delta_offsets)

    if (-1 * min_offset) > min_idx:
        early_lift = min_idx
    else:
        early_lift = 0

    min_idx = min_offset * -1
    crop_first = 0
    # crop_last = max_idx + max_offset
    crop_last = max([max_idx, max_idx + max_offset])

    if is_empty is True:
        early_lift = 10000

    return min_idx, max_idx, crop_first, crop_last, early_lift


def parse_matfile(opts):
    """Parse hantman mouse matfile data.

    Given an opts dict (created from setup_opts), process a matfile and save
    it as a pickle in outname.
    """
    matfile = sio.loadmat(opts['filename'])

    # the labels that I care about are:
    # Lift_labl_t0sPos
    # Handopen_labl_t0sPos
    # Grab_labl_t0sPos
    # Sup_labl_t0sPos
    # Atmouth_labl_t0sPos
    # Chew_labl_t0sPos

    # associated labelling
    # nothing = 0
    # hand open = 1
    # grab = 2
    # suppinate = 3
    # at mouth = 4
    # chew = 5
    label_names = ['Lift_labl_t0sPos', 'Handopen_labl_t0sPos',
                   'Grab_labl_t0sPos', 'Sup_labl_t0sPos',
                   'Atmouth_labl_t0sPos', 'Chew_labl_t0sPos']

    # each index in this mat file is an experiment (video). For each video,
    # create feature and label matrix. For label information, check the
    # rawdata field. For the trajectory data, check trxdata.
    num_experiments = matfile['trxdata'].size
    all_feats = []
    all_labels = []
    all_experiments = []
    all_num_frames = []
    all_exp_dir = []
    all_crops = []
    all_frame_idx = []
    # all_crop_first = []
    # all_crop_last = []
    # all_org_frames = []
    frame_offsets = opts['frame_offsets']  # = [-20, -10, -5, 0, 5, 10, 20]
    delta_offsets = opts['delta_offsets']  # [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # window_offsets = [0]
    # delta_offsets = [0]

    # keep track of skipped files
    early_lift_exps = []
    no_label_exps = []

    # for i in range(num_experiments):
    # sub_exp = range(0, 20) + range(500, 520)  # + range(600, 620)
    sub_exp = list(range(num_experiments))
    for i in sub_exp:
        print("%d of %d" % (i, num_experiments))
        # print i
        # if i == 6:
        print("\t%s" % matfile['rawdata'][i][0]['exp'][0])
        exp_name, exp_dir, labels, features, crops, frame_idx, early_lift =\
            process_video(
                opts, label_names, matfile['rawdata'][i][0],
                matfile['trxdata'][0][i],
                frame_offsets, delta_offsets)

        # if 'M134_20150506_v013' == exp_name or early_lift == 10000:
        #     import pdb; pdb.set_trace()
        print("\t%d" % matfile['trxdata'][0][i]['x1'].size)
        if early_lift == 0:
            all_experiments.append(exp_name)
            all_exp_dir.append(exp_dir)
            all_labels.append(labels)
            all_feats.append(features)
            all_crops.append(crops)
            all_frame_idx.append(frame_idx)
            all_num_frames.append(matfile['trxdata'][0][i]['x1'].size)
        else:
            print("\texp_name: %s" % exp_name)
            print("\tfirst lift: %d" % early_lift)
            if early_lift < 10000:
                early_lift_exps.append(exp_name)
            else:
                no_label_exps.append(exp_name)

    # END for i in range(num_experiments):

    # write the examples to file
    early_filename = os.path.join(opts["out_dir"], "early_behavior.txt")
    with open(early_filename, "w") as outfile:
        for early_name in early_lift_exps:
            outfile.write("%s\n" % early_name[0])
    no_label_filename = os.path.join(opts["out_dir"], "no_label_names.txt")
    no_label_exps.sort()
    with open(no_label_filename, "w") as outfile:
        for no_label_name in no_label_exps:
            outfile.write("%s\n" % no_label_name[0])
    # now write the experiments that were being used to file.
    used_exps_filename = os.path.join(opts["out_dir"], "used_exps.txt")
    # need sort the list of experiments, but all_experiments needs to match
    # the order of the rest of the all_* lists.
    sorted_exp_names = [tmp_name for tmp_name in all_experiments]
    sorted_exp_names.sort()
    with open(used_exps_filename, "w") as outfile:
        for used_exp in sorted_exp_names:
            outfile.write("%s\n" % used_exp[0])

    data = dict()
    data['exp'] = all_experiments
    data['exp_dir'] = all_exp_dir
    data['label_names'] = label_names
    data['features'] = all_feats
    data['labels'] = all_labels
    # data['crop_first'] = all_crop_first
    # data['crop_last'] = all_crop_last
    data['crops'] = all_crops
    data['num_frames'] = all_num_frames
    data['frame_idx'] = all_frame_idx
    # data['org_frames'] = all_org_frames

    return data

if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    # create the output directory
    # paths.setup_output_space(opts['out_dir'])
    paths.create_dir(opts['out_dir'])
    paths.save_command(opts['out_dir'])

    # log the git information
    git_helper.log_git_status(
        os.path.join(opts['out_dir'], '00_git_status.txt'))
    data = parse_matfile(opts)

    outname = os.path.join(opts['out_dir'], 'data.npy')
    joblib.dump(data, outname)

    # create the hdf5 verson of the data