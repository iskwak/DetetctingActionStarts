from __future__ import print_function, division

import os
import h5py
# import numpy


def process_hdf(hdf_filename, hoghof_dir, exp_dir):
    # load both exp files.
    with h5py.File(os.path.join(hoghof_dir, hdf_filename), 'r') as hoghof:
        if not os.path.isfile(os.path.join(exp_dir, hdf_filename)):
            print("%s doesn't exist" % hdf_filename)
            return

        with h5py.File(os.path.join(exp_dir, hdf_filename), 'a') as base_exp:
            # compare the number of frames in each.
            if base_exp['labels'].shape[0] > hoghof['hoghof'].shape[0]:
                print("%s has more frames in the base_exp")

            # print(base_exp['labels'].shape)
            # print(hoghof['hoghof'].shape)
            num_frames = base_exp['labels'].shape[0]
            # print("%d, %d" % (hoghof['hoghof'].value[:num_frames].shape[0], hoghof["hoghof"].shape[0]))
            base_exp['hoghof'] = hoghof['hoghof'][:num_frames]


def get_experiments(dirname):
    all_files = os.listdir(dirname)
    all_files.sort()

    all_files = [
        filename for filename in all_files
        if not os.path.islink(os.path.join(dirname, filename)) and
        not os.path.isdir(os.path.join(dirname, filename))
    ]
    # print(all_files[-1])

    return all_files


def main():
    hoghof_dir = '/media/drive1/data/hantman_processed/hdf5_data/exps'
    # exp_dir = '/nrs/branson/kwaki/data/20180708_base_hantman/exps'
    exp_dir = '/nrs/branson/kwaki/data/20180729_base_hantman/exps'

    org_files = os.listdir(hoghof_dir)
    org_files.sort()

    exp_files = get_experiments(exp_dir)
    # exp_files.sort()
    # make sure there is a hoghof exp file for each exp_file.
    diff = [item for item in exp_files if item not in org_files]

    if diff:
        # if the list isn't empty, then there are some features missing.
        print("Missing some features")
        print(diff)
        return

    # loop over files.
    for filename in exp_files:
        process_hdf(filename, hoghof_dir, exp_dir)


if __name__ == "__main__":
    main()
