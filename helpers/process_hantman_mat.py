"""Get some stats on the hantman matfile."""
import argparse
import numpy
import pickle


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
    args = parser.parse_args()

    opts['filename'] = args.filename
    return opts


def convert_data(data):
    """Convert the spike style data to timing style data."""
    # create a new label matrix with timing information rather than spikes
    num_labels = len(data['label_names'])
    num_experiments = len(data['exp'])

    all_timings = []
    for i in range(num_experiments):
        num_frames = data['num_frames'][i]
        timings = numpy.zeros((num_frames, num_labels))

        curr_labels = data['labels'][i]

        # find the first behavior
        prev_idx = -1
        prev_where_label = -1
        for j in range(num_frames):
            where_label = numpy.where(curr_labels[j])
            if where_label[0].size > 0:
                prev_idx = j
                prev_where_label = where_label
                timings[j][where_label] = 1
                break
        if prev_idx < num_frames:
            # if a behavior is found continue processing
            for j in range(prev_idx, num_frames):
                where_label = numpy.where(curr_labels[j])
                if where_label[0].size > 0:
                    timings[j][where_label] = 1
                    prev_where_label = where_label
                else:
                    timings[j][prev_where_label] = 1 + \
                        timings[j - 1][prev_where_label]

        all_timings.append(timings)
    return all_timings


def process_matfile(opts):
    """Process hantman mouse pickled data.

    Given an opts dict (created from setup_opts), process a matfile and
    figure out a few things, such as number of experiments with no lift
    labeled.
    """
    with open(opts['filename'], 'r') as infile:
        data = pickle.load(infile)

        # for the hantman data, it would be nice to know a few things, such
        # as the frame gap between behaviors.
        # list of things:
        # how often a label appears (and doesn't appear) per experiment
        # the gap between labels (from the last labeled behavior
        # gap between pairs of labels
        # mean feature locations.
        num_exp = len(data['exp'])
        num_labels = len(data['label_names'])
        # feature_dim = data['features'][0].shape[1]

        label_counts = numpy.zeros((num_exp, num_labels))
        label_gaps = []

        # label pairs:
        # lift -> hand
        # hand -> grab
        # grab -> sup
        # sup -> at mouth
        # at mouth -> chew
        # between_label_gaps = numpy.zeros((num_exp, num_labels - 1))

        # might be useful to get feature locations. (x1,y1,x2,y2)
        # list pre-initialization in python is strange...
        # feature_locations = [None] * num_labels
        feature_locations = dict()
        for i in range(num_labels):
            feature_locations[i] = []

        # finally the last thing to look for is the number of experiments
        # where the order is incorrect (ie lift is not first, handopen is
        # not second)
        # out_order_labels = numpy.zeros((num_exp, num_labels))

        # look for the gaps between behaviors.
        for i in range(num_exp):
            label_idx = []

            # for each experiment, figure out how often each behavior occurs
            for j in range(num_labels):
                # per exp label count
                label_counts[i, j] = data['labels'][i][:, j].sum()
                label_idx.append(numpy.nonzero(data['labels'][i][:, j])[0])

                if label_idx[j].any():
                    curr_features = data['features'][i][label_idx[j], :]
                    feature_locations[j].append(curr_features[0])

            # figure out the gaps.
            all_idx = numpy.concatenate(label_idx)
            label_gaps.append(numpy.diff(all_idx))

        label_gaps = numpy.concatenate(label_gaps)
        print("label_gaps: ", label_gaps.mean(), label_gaps.std())
        for i in range(num_labels):
            print(data['label_names'][i])
            print("\tlabel_counts: ", label_counts[:, i].mean(), \
                label_counts[:, i].std())
            curr_features = numpy.vstack(feature_locations[i])
            print("\t", curr_features.shape)
            print("\tlocation: ", curr_features.mean(axis=0), \
                curr_features.std(axis=0))


if __name__ == "__main__":
    opts = create_opts()
    opts = setup_opts(opts)

    process_matfile(opts)
