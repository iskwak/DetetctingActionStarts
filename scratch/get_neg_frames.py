import h5py
import numpy

files = [
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_train.hdf5',
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_valid.hdf5',
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_test.hdf5'
]


num_neg = 0
for filename in files:
    with h5py.File(filename, "r") as h5data:
        print(filename)
        exp_names = h5data["exp_names"].value
        for exp_name in exp_names:
            labels = h5data["exps"][exp_name]["labels"].value
            pos = numpy.argwhere(labels == 1).shape[0]
            num_neg = num_neg + (labels.shape[0] - pos)
        print(num_neg)
