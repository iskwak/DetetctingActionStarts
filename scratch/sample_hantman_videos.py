"""Test video sampling... Need this because storing frames for MPII doesn't work."""
import numpy
# import theano

# import threading
# import time
import time
import h5py
from helpers.videosampler import HantmanVideoFrameSampler
from helpers.videosampler import HantmanVideoSampler
import torch
# from helpers.videosampler import VideoHDFSampler
# from helpers.videosampler import VideoHDFFrameSampler

# python 2 vs 3 stuff...
# import sys
# if sys.version_info[0] < 3:
#     import Queue as queue
# else:
#     import queue


def main():
    video_path = '/media/drive1/data/hantman_pruned'
    base_hdf = '/media/drive1/data/hantman_processed/20180605_base_hantman/hantman_test.hdf5'
    rng = numpy.random.RandomState(123)
    mini_batch = 10

    torch.cuda.set_device(2)

    with h5py.File(base_hdf, 'r') as hdf5_data:
        sampler = HantmanVideoFrameSampler(
            rng, hdf5_data, video_path, mini_batch,
            frames=[-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16],
            use_pool=False, gpu_id=-1)
        # sampler = HantmanVideoFrameSampler(
        #     rng, hdf5_data, video_path, mini_batch, frames=[0],
        #     use_pool=True, gpu_id=2)
        # sampler = HantmanVideoSampler(
        #     None, hdf5_data, video_path, seq_len=-1,
        #     use_pool=False, gpu_id=-1)
        sampler.reset()
        exp_names = hdf5_data["exp_names"].value
        exp_names.sort()

        print(sampler.num_batch)
        tic = time.time()
        # seen = []
        # sizes = []
        for i in range(sampler.num_batch):
            print("i: %d" % i)
            data = sampler.get_minibatch()
            # seen.append(data[-1][0])
            # sizes.append(data[0].shape[0])
            print("\t%d" % data[0].shape[0])
        # import pdb; pdb.set_trace()

        sampler.reset()
        # again!
        for i in range(sampler.num_batch):
            print("i: %d" % i)
            data = sampler.get_minibatch()
        print(time.time() - tic)

        # import pdb; pdb.set_trace()
        print("hi")

    return


if __name__ == "__main__":
    main()
