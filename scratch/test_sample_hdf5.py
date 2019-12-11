"""Test HDF file sampling."""
import numpy
# import theano

# import threading
# import time
import time
import h5py
from helpers.videosampler import HDF5Sampler
import torch


def main():
    base_hdf = '/nrs/branson/kwaki/data/20180619_jigsaw_base/Suturing_1_Out_train.hdf5'
    rng = numpy.random.RandomState(123)
    mini_batch = 34

    torch.cuda.set_device(2)

    feat_keys = [
        ['split_1', 'hog'],
        ['split_1', 'hof']
    ]

    with h5py.File(base_hdf, 'r') as hdf5_data:
        sampler = HDF5Sampler(
            None, hdf5_data, 2, feat_keys, seq_len=3000,
            use_pool=False
        )
        # moo = sampler._get_field('Suturing_I004', feat_keys[0])
        sampler.reset()
        tic = time.time()
        for i in range(sampler.num_batch):
            blob = sampler.get_minibatch()
            print(blob['names'])
        print(time.time() - tic)

        # sampler.reset()
        # tic = time.time()
        # for i in range(sampler.num_batch):
        #     sampler.get_minibatch()
        #     # print(blob['names'])
        # print(time.time() - tic)

        # sampler.reset()
        # tic = time.time()
        # for i in range(sampler.num_batch):
        #     sampler.get_minibatch()
        #     # print(blob['names'])
        # print(time.time() - tic)
        # import pdb; pdb.set_trace()
        print('hi')


if __name__ == "__main__":
    main()
