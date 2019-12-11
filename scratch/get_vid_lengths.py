import h5py

data_files = [
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_train.hdf5',
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_valid.hdf5',
    '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M147_test.hdf5'
]

vid_lengths = []
moo = [0, 0, 0, 0, 0, 0]
for data_file in data_files:
    with h5py.File(data_file, "r") as data:
        for exp in data["exp_names"]:
            labels = data["exps"][exp]["labels"].value
            vid_lengths.append(labels.shape[0])
            for i in range(labels.shape[1]):
                moo[i] += labels[:, i].sum()
import pdb; pdb.set_trace()
