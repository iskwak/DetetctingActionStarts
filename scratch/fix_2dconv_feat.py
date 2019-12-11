import os
import h5py


def fix_features(h5, exp_name):
    feature_folders = {
        'M134': '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM134',
        'M147': '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM147',
        'M173': '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM173',
        'M174': '/nrs/branson/kwaki/data/features/hantman_2dconv/20181103_splitM174'
    }

    # if 2dconv is in the h5data, delete it.
    # if "2dconv" in h5.data["exps"][curr_exp]:
    #     # import pdb; pdb.set_trace()
    #     del h5.data["exps"][curr_exp]["2dconv"]

    # for each split folder relink the features with the split name as part of
    # the key.
    for split_key in feature_folders.keys():
        # setup the split folder/key
        split_folder = feature_folders[split_key]
        split_folder = os.path.basename(split_folder)

        key_name = split_key + "/2dconv"
        h5["exps"][exp_name][key_name] = h5py.ExternalLink(
            os.path.join(os.path.basename(split_folder), exp_name),
            "/2dconv"
        )


def process_file(filename):
    with h5py.File(filename, "a") as h5:
        exp_names = h5["exp_names"].value
        for exp_name in exp_names:
            fix_features(h5, exp_name)


def main():
    # just use one set of h5 files, they should all touch the same files.
    filenames = [
        '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_train.hdf5',
        '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_test.hdf5',
        '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_valid.hdf5'
    ]

    for filename in filenames:
        process_file(filename)


if __name__ == "__main__":
    main()
