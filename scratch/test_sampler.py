"""Script to test the hantman sampler."""
from helpers.hantman_sampler import HantmanFrameSeqSampler
import h5py
import numpy
import time

# train_fname = ("/media/drive1/data/hantman_processed/20170827_vgg/"
#                "one_mouse_multi_day_train.hdf5")
# test_fname = ("/media/drive1/data/hantman_processed/20170827_vgg/"
#               "one_mouse_multi_day_test.hdf5")
train_fname = ("/media/drive3/kwaki/data/hantman_processed/20180305_imgs/"
               "one_mouse_multi_day_test.hdf5")
frame_path = ("/media/drive1/data/hantman_frames")

num_epochs = 100
mini_batch = 10
seq_len = 1500
rng = numpy.random.RandomState()
with h5py.File(train_fname, "r") as train_data:
    train_sampler = HantmanFrameSeqSampler(
        rng, train_data, frame_path, seq_len, mini_batch, use_pool=True,
        max_workers=1, max_queue=10
    )
    exp_names = []
    tic = time.time()
    print("going through 10 batches, with pool")
    for i in range(10):
        print(i)
        batch = train_sampler.get_minibatch()
        exp_names.append(batch["exp_names"])
        # print("huh?")
    print("Time: %f" % (time.time() - tic))

    # check that the first 100 exp names were seen
    t1 = numpy.setdiff1d(
        numpy.array(exp_names).flatten().sort(),
        train_sampler.exp_names[:100].sort()
    )
    t2 = numpy.setdiff1d(
        numpy.array(train_sampler.exp_names[:100]).sort(),
        numpy.array(exp_names).flatten().sort()
    )
    if t1.size != 0 or t2.size != 0:
        print("seen examples doesn't make sense")
        import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()

    # train_sampler = HantmanFrameSeqSampler(
    #     rng, train_data, frame_path, seq_len, mini_batch, use_pool=False
    # )
    # print("going through 10 batches, no pool")
    # tic = time.time()
    # for i in range(10):
    #     print(i)
    #     batch = train_sampler.get_minibatch()
    # print("Time: %f" % (time.time() - tic))

    # import pdb; pdb.set_trace()
    print("hi")
    # del train_sampler
    # train_sampler.close_workers()
    print("moo")
print("?")

# test the sequence loader.

 



# train_fname = ("/media/drive1/data/hantman_processed/20170814_vgg/"
#                "data.hdf5")
# train_fname = (
#     "C:/Users/ikwak/Desktop/research/sample_data/"
#     "one_mouse_one_day_train.hdf5"
# )
# test_fname = (
#     "C:/Users/ikwak/Desktop/research/sample_data/"
#     "one_mouse_one_day_test.hdf5"
# )
# num_epochs = 100
# mini_batch = 100
# rng = numpy.random.RandomState()

# with h5py.File(train_fname, "r") as train_data:
#     train_sampler = HantmanFrameSampler(
#         rng, train_data, mini_batch, ["img_side", "img_front"], use_pool=False,
#         max_workers=4, max_queue=10
#     )
#     # print train_sampler.get_mini_batch
#     # train_sampler = HantmanFrameSampler()
#     print("going through 10 batches, no pool")
#     tic = time.time()
#     for i in range(10):
#         print(i)
#         batch = train_sampler.get_minibatch()
#     print("Time: %f" % (time.time() - tic))

#     train_sampler = HantmanFrameSampler(
#         rng, train_data, mini_batch, ["img_side", "img_front"], use_pool=True,
#         max_workers=9, max_queue=10
#     )
#     tic = time.time()
#     print("going through 10 batches, with pool")
#     for i in range(10):
#         print(i)
#         batch = train_sampler.get_minibatch()
#         # print("huh?")
#     print("Time: %f" % (time.time() - tic))
#     # import pdb; pdb.set_trace()
#     print("hi")
#     # del train_sampler
#     train_sampler.close_workers()
#     print("moo")
# print("?")

# test the sequence loader.
