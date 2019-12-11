"""Convert matlab mouse data into a pickle."""
import scipy.io as sio
import argparse
import numpy
import pickle
# based on parse_hantman_mat, but tweaked to make it easier to create data
# pickles with various feature setups (such as concatenating multiple frames of
# features together).


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts['filename'] = ''
    opts['outname'] = ''
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
    parser.add_argument('-o', '--outname', required=True, type=str,
                        help='output picke file name')
    args = parser.parse_args()

    opts['filename'] = args.filename
    opts['outname'] = args.outname
    return opts


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
    all_crop_first = []
    all_crop_last = []
    all_org_frames = []
    for i in range(num_experiments):
        num_frames = matfile['trxdata'][0][i]['x1'].size
        all_org_frames.append(num_frames)
        all_experiments.append(matfile['rawdata'][i][0]['exp'])

        # before taking the features, some "pruning" of the video will
        # need to be done. Because the video may end up having a lot of
        # dead time, cut out the "middle" frames. Crop the video with
        # 30 frames before the first label and 30 frames after the last
        # label.

        # the first labeled frame is always a lift action. So first check
        # the lift.
        min_idx = num_frames
        max_idx = 0
        for j in range(len(label_names)):
            t0s_str = label_names[j]
            t0s = matfile['rawdata'][i][0][t0s_str]
            # print t0s
            if t0s[0].size > 0 and t0s[0][0] < min_idx:
                min_idx = t0s[0][0]
            if t0s[0].size > 0 and t0s[0][-1] > max_idx:
                max_idx = t0s[0][-1]

        # if min_idx didn't change, then this means that there were no
        # behaviors.
        if min_idx == num_frames:
            min_idx = 0
        if max_idx == 0:
            max_idx = num_frames

        # print num_frames
        # print i, ": ", matfile['rawdata'][i][0]['exp']
        # print "\t", min_idx
        # print "\t", max_idx

        crop_first = max(min_idx - 30, 0)
        crop_last = min(max_idx + 30, num_frames)
        all_crop_first.append(crop_first)
        all_crop_last.append(crop_last)
        # print "\t", crop_first
        # print "\t", crop_last
        num_frames = crop_last - crop_first
        # print "\t", num_frames
        all_num_frames.append(num_frames)
        labels = numpy.zeros((num_frames, len(label_names)),
                             dtype=numpy.float32)

        # there is no need to figure out the last frame number because the
        # labels are unaffected by early termination of a video.

        # create a label matrix
        labels = numpy.zeros((num_frames, len(label_names)),
                             dtype=numpy.float32)

        for j in range(len(label_names)):
            # assuming that the strings in label_base are the correct base
            # for fields in the raw data, just add t0sPos and t1sPos.
            t0s_str = label_names[j]
            t0s = matfile['rawdata'][i][0][t0s_str]
            # include the offset
            t0s = t0s - crop_first

            for k in range(t0s.size):
                labels[t0s[0][k], j] = 1

        x1 = matfile['trxdata'][0][i]['x1'][crop_first:crop_last]
        y1 = matfile['trxdata'][0][i]['y1'][crop_first:crop_last]
        x2 = matfile['trxdata'][0][i]['x2'][crop_first:crop_last]
        y2 = matfile['trxdata'][0][i]['y2'][crop_first:crop_last]
        # additionally, create feature deltas
        del_x1 = matfile['trxdata'][0][i]['x1'][crop_first:crop_last] -\
            matfile['trxdata'][0][i]['x1'][crop_first - 1:crop_last - 1]
        del_y1 = matfile['trxdata'][0][i]['y1'][crop_first:crop_last] -\
            matfile['trxdata'][0][i]['y1'][crop_first - 1:crop_last - 1]
        del_y2 = matfile['trxdata'][0][i]['x2'][crop_first:crop_last] -\
            matfile['trxdata'][0][i]['x2'][crop_first - 1:crop_last - 1]
        del_x2 = matfile['trxdata'][0][i]['y2'][crop_first:crop_last] -\
            matfile['trxdata'][0][i]['y2'][crop_first - 1:crop_last - 1]

        # create the "feature" matrix
        # features = numpy.zeros((num_frames, 4), dtype=numpy.float32)
        features = numpy.concatenate((x1, y1, x2, y2), axis=1)

        all_labels.append(labels)
        all_feats.append(features)

    data = dict()
    data['exp'] = all_experiments
    data['label_names'] = label_names
    data['features'] = all_feats
    data['labels'] = all_labels
    data['crop_first'] = all_crop_first
    data['crop_last'] = all_crop_last
    data['num_frames'] = all_num_frames
    data['org_frames'] = all_org_frames

    return data


def create_feature_mat(opts, data):
    """Modify the features."""
    return data

if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    data = parse_matfile(opts)

    # after creating the main data features, create the desired variation
    data = create_feature_mat(opts, data)

    if opts['outname']:
        with open(opts['outname'], 'wb') as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)
