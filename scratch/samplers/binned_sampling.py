"""Test the new sampling."""
from __future__ import print_function, division
# import os
import time
import sys
import numpy as np
import h5py
import sys
import helpers.videosampler
from helpers.videosampler import HDF5Sampler
from helpers.videosampler import HDF5CachedSampler
from helpers.videosampler import HDF5BinnedSampler


def test_sampler(h5data):
    feat_keys = [
        ["canned_i3d_rgb_64"]
    ]
    seq_len = -1
    cuda_device = 1
    rng = np.random.RandomState()

    sampler = HDF5BinnedSampler(
        rng, h5data, 4,
        feat_keys, seq_len=seq_len,
        use_pool=True, gpu_id=cuda_device,
        use_cached=False)

    if sampler.batch_idx.empty():
        sampler.reset()

    timings = [0, 0, 0, 0]
    for i in range(sampler.num_batch):
        # get data blob
        tic = time.time()
        blob = sampler.get_minibatch()
        print(blob["names"])
    # import pdb; pdb.set_trace()

    # exp_names = h5data["exp_names"][()]
    # feat_dim = h5data["exps"][exp_names[0]][feat_keys[0][0]].shape[1]
    # vid_lengths = []
    # for i in range(len(exp_names)):
    #     vid_lengths.append(
    #         h5data["exps"][exp_names[i]][feat_keys[0][0]].shape[0]
    #     )
    # sort_idx = np.flip(np.argsort(vid_lengths))
    # vid_lengths = np.asarray(vid_lengths)[sort_idx]
    # exp_names = exp_names[sort_idx]

    # features = []
    # sizes = []
    # for i in range(len(exp_names)):
    #     print("%d of %d" % (i, len(exp_names)))
    #     print("\t%fgb" % (1.0 * sum(sizes) / 1000 / 1000 / 1000))
    #     temp = h5data["exps"][exp_names[i]][feat_keys[0][0]][()].astype("float32")
    #     features.append(temp)
    #     sizes.append(
    #         sys.getsizeof(temp)
    #     )
    # import pdb; pdb.set_trace()

def main(argv):
    print(argv)

    train_file  = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5"
    # test_file = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/test.hdf5"
    test_file = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5"

    # full_tic = time.time()
    # with h5py.File(train_file, "r") as train_data:
    with h5py.File(test_file, "r") as test_data:
        test_sampler(test_data)


if __name__ == "__main__":
    main(sys.argv)
