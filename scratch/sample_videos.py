"""Test video sampling... Need this because storing frames for MPII doesn't work."""
import numpy
# import theano

# import threading
# import time
import time
import h5py
# from helpers.videosampler import VideoFrameSampler
from helpers.videosampler import VideoSampler
# from helpers.videosampler import VideoHDFSampler
# from helpers.videosampler import VideoHDFFrameSampler

# python 2 vs 3 stuff...
# import sys
# if sys.version_info[0] < 3:
#     import Queue as queue
# else:
#     import queue


def main():
    # video_path = '/nrs/branson/kwaki/data/20180501_jigsaw_base/videos/'
    # base_hdf = '/nrs/branson/kwaki/data/20180424_jigsaw/data.hdf5'
    # base_hdf = '/media/drive3/kwaki/data/jigsaw/20180424_jigsaw_base/data.hdf5'
    # base_hdf = '/nrs/branson/kwaki/data/20180501_jigsaw_base/data.hdf5'
    # base_hdf = '/nrs/branson/kwaki/data/20180501_jigsaw_base/debug_Knot_Tying_1_Out_test.hdf5'
    # base_hdf = '/nrs/branson/kwaki/data/20180529_jigsaw_base/debug_Knot_Tying_1_Out_test.hdf5'
    # base_hdf = '/nrs/branson/kwaki/data/20180529_jigsaw_base/Knot_Tying_1_Out_train.hdf5'
    video_path = '/nrs/branson/kwaki/data/hantman_pruned/'
    base_hdf = '/nrs/branson/kwaki/data/20180729_base_hantman/data.hdf5'
    rng = numpy.random.RandomState(123)
    mini_batch = 10

    with h5py.File(base_hdf, 'r') as hdf5_data:
        # sampler = VideoFrameSampler(
        #     rng, hdf5_data, video_path, mini_batch,
        #     frames=[-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16],
        #     use_pool=False, gpu_id=0)
        # sampler = VideoFrameSampler(
        #     rng, hdf5_data, video_path, mini_batch, frames=[0],
        #     use_pool=False)
        sampler = VideoSampler(
            None, hdf5_data, video_path, seq_len=-1,
            use_pool=False, gpu_id=-1)
        # sampler = VideoHDFSampler(
        #     None, hdf5_data, seq_len=3000,
        #     use_pool=False, gpu_id=-1)
        # sampler = VideoHDFFrameSampler(
        #     rng, hdf5_data, mini_batch, frames=[-10, -5, -1, 0],
        #     use_pool=False, gpu_id=0)
        # sampler.reset()
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

# sampler thoughts
# find the positive samples.
# cycle through them
# enumerate all the



# other sampler?
# idx of videos?


# class VideoHDFSampler(object):
#     """VideoHDFSampler

#     Samples random videos, assumes that there is a frames key that has all
#     the video frames in the HDF5 file.

#     If use_pool is true, the sampler will be multithreaded and maybe faster.
#     Unfortunately there's no guarentee that the pooled data loader is faster.
#     """
#     def __init__(self, rng, hdf_data, max_workers=2, seq_len=-1, max_queue=5,
#                  use_pool=False, gpu_id=-1, normalize=None, channels=3,
#                  width=224, height=224):
#         self.rng = rng
#         self.data = hdf_data
#         self.use_pool = use_pool
#         self.use_gpu = gpu_id
#         self.exp_names = self.data["exp_names"].value
#         self.exp_names.sort()
#         self.label_dims =\
#             self.data["exps"][self.exp_names[0]]["labels"].shape[2]
#         self.channels = channels
#         self.width = width
#         self.height = height
#         self.seq_len = seq_len

#         self.num_exp = len(self.exp_names)
#         self.exp_ids = numpy.asarray(range(self.num_exp))
#         self.num_behav =\
#             self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

#         # compute the number of "batches" that can be created with the number
#         # of positive examples.
#         # Ignore remainders? Assume leftovers will be seen by the network
#         # eventually.
#         self.num_batch = self.num_exp
#         self.batch_idx = queue.Queue(self.num_batch)
#         # setup the queue.
#         # self.reset()

#         # setup the preprocessing
#         if normalize is None:
#             # Scale doesn't seem to be taking in a sequence, even tho the
#             # help says it should...
#             self.preproc = transforms.Compose([
#                 transforms.Scale(self.width),
#                 # transforms.Scale([self.width, self.height]),
#                 # transforms.CenterCrop(224),
#                 transforms.ToTensor()
#             ])
#         else:
#             self.preproc = transforms.Compose([
#                 transforms.Scale(self.width),
#                 # transforms.Scale([self.width, self.height]),
#                 # transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize
#             ])

#         # initialize the workers if necessary.
#         if self.use_pool is True:
#             self._pool = DataLoader.DataLoaderPool(
#                sampler=self.batch_sampler, max_workers=max_workers,
#                max_queue=max_queue
#             )

#     def reset(self):
#         """Reset the sampler."""
#         if self.rng is not None:
#             self.rng.shuffle(self.exp_ids)
#         for i in range(self.num_batch):
#             self.batch_idx.put(self.exp_ids[i])
#         # print("\treset!")

#     def get_minibatch(self):
#         """Get a minibatch of data."""
#         if self.use_pool is True:
#             minibatch = self._pool.get()
#         else:
#             minibatch = self.batch_sampler()

#         # if self.batch_idx.empty():
#         #     # the batch_idx is empty. an "epoch" has passed. resample the
#         #     # positive index order.
#         #     print("\treset")
#         #     # self.reset()

#         return minibatch

#     def get_frames(self, exp_idx):
#         # helper to deal with the frame offsets.
#         exp_name = self.exp_names[exp_idx]
#         exp = self.data["exps"][exp_name]
#         # video_name = self.data["exps"][exp_name]["video_name"].value
#         # print(exp_name)

#         # pre-allocate the space
#         np_labels = exp["org_labels"].value
#         num_frames = exp["frames"].shape[0]
#         if self.seq_len > -1:
#             frames = torch.zeros(
#                 self.seq_len, self.channels, self.width, self.height)
#             labels = torch.zeros(self.seq_len, self.label_dims)
#             num_frames = numpy.min([num_frames, self.seq_len])
#         else:
#             frames = torch.zeros(
#                 num_frames, self.channels, self.width, self.height)
#             labels = torch.zeros(num_frames, self.label_dims)
#         # the number of frames to process is the minimum of the desired
#         # sequence length and the number of frames in the video.
#         # tic = time.time()

#         tic = time.time()
#         frames = torch.Tensor(exp["frames"].value[:num_frames, :])
#         print(time.time() - tic)
#         import pdb; pdb.set_trace()

#         # get the labels
#         num_labels = np_labels.shape[0]
#         num_labels = numpy.min([num_frames, num_labels])
#         # labels = torch.zeros(num_frames, self.label_dims)
#         labels[:num_labels, :] = torch.from_numpy(np_labels)[:num_labels, :]

#         # cap.release()

#         return frames, labels

#     def batch_sampler(self):
#         batch_id = self.batch_idx.get()
#         # print "worker %d: loaded data, waiting to put..." % self.threadid
#         # now actually get the batch
#         idx_range = list(range(
#             batch_id,
#             batch_id + 1))

#         # # preallocate the space
#         # features = torch.zeros(
#         #     self.seq_len, self.channels, self.width, self.height)
#         # labels = torch.zeros(self.seq_len, self.label_dims)

#         # get the exps
#         sampled_exps = self.exp_ids[idx_range]

#         cur_feats, cur_labels =\
#             self.get_frames(sampled_exps[0])
#         features = cur_feats
#         labels = cur_labels
#         # import pdb; pdb.set_trace()
#         # feat_var = [
#         #     Variable(var, requires_grad=True) for var in features
#         # ]
#         # label_var = [Variable(labels, requires_grad=False)]
#         feat_var = [features]
#         label_var = [labels]

#         inputs = feat_var + label_var
#         if self.use_gpu >= 0:
#             inputs = [
#                 input.cuda(self.use_gpu) for input in inputs
#             ]
#         inputs = inputs + [self.exp_names[sampled_exps]]
#         return inputs

#     def preprocess_img(self, img):
#         """Apply pytorch preprocessing."""
#         # split the image into two frames
#         img = cv2.resize(img, (self.width, self.height))
#         pil_img = PIL.Image.fromarray(img)
#         tensor_img = self.preproc(pil_img)

#         return tensor_img


# class VideoHDFFrameSampler(object):
#     """VideoFrameSampler

#     Samples random frames from videos.

#     If use_pool is true, the sampler will be multithreaded and maybe faster.
#     Unfortunately there's no guarentee that the pooled data loader is faster.
#     """
#     def __init__(self, rng, hdf_data, mini_batch, max_workers=2,
#                  frames=[0], max_queue=5, use_pool=False, gpu_id=-1,
#                  normalize=None, channels=3, width=224, height=224):
#         self.rng = rng
#         self.data = hdf_data
#         self.use_pool = use_pool
#         self.use_gpu = gpu_id
#         self.frames = frames
#         self.exp_names = self.data["exp_names"]
#         self.label_dims =\
#             self.data["exps"][self.exp_names[0]]["labels"].shape[2]
#         self.channels = channels
#         self.width = width
#         self.height = height

#         self.mini_batch = mini_batch
#         # Because positives are so rare, we'll want to actively find positive
#         # samples for the minibatch. This variable will deterimine how many
#         # positive samples are needed per patch.
#         self._pos_batch_size = int(mini_batch * 4 / 5)
#         self._neg_batch_size = mini_batch - self._pos_batch_size

#         # self.num_exp = 3
#         self.num_exp = len(self.exp_names)
#         self.num_behav =\
#             self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

#         self.pos_batch_idx = 0
#         # the following values will be setup by _find_positives()
#         self.pos_locs = []
#         self.num_pos = 0
#         self.num_neg = 0
#         # First column is the experiment index, second the frame number for
#         # that experiment and then the third column is the behavior index.
#         self.pos_idx = []
#         # Each index is a scalar that is between 0 and number of negative
#         # samples. The scalar represents a frame that indexes into the
#         # self.neg_bouts variable.
#         # For example, if self.neg_bouts[0] = [0, 0, 120, 120] and
#         # self.neg_bouts[1] = [0, 130, 146, 16], and self.neg_idx[0] = 122.
#         # Then self.neg_idx[i] represents frame 132 of experiment 0.
#         self.neg_idx = []
#         # Find the positive examples (and figure out the negative bouts).
#         self._find_positives()

#         # compute the number of "batches" that can be created with the number
#         # of positive examples.
#         # Ignore remainders? Assume leftovers will be seen by the network
#         # eventually.
#         self.num_batch = int(self.num_pos / self._pos_batch_size)
#         self.batch_idx = queue.Queue(self.num_batch)
#         # setup the queue.
#         # self.reset()

#         # setup the preprocessing
#         # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                                  std=[0.229, 0.224, 0.225])
#         if normalize is None:
#             self.preproc = transforms.Compose([
#                 transforms.Scale(224),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor()
#                 # normalize
#             ])
#         else:
#             self.preproc = transforms.Compose([
#                 transforms.Scale(224),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 normalize
#             ])

#         # initialize the workers if necessary.
#         if self.use_pool is True:
#             self._pool = DataLoader.DataLoaderPool(
#                sampler=self.batch_sampler, max_workers=max_workers,
#                max_queue=max_queue
#             )

#     def reset(self):
#         """Reset the sampler."""
#         self.rng.shuffle(self.pos_idx)
#         for i in range(self.num_batch):
#             self.batch_idx.put(i)

#         # create a random sequence of negative
#         self.neg_idx = self.rng.choice(
#             self.num_neg,
#             (self._neg_batch_size * self.num_batch),
#             replace=False)
#         # self.half_batch_idx = 0
#         # print("\treset!")

#     def get_minibatch(self):
#         """Get a minibatch of data."""
#         if self.use_pool is True:
#             minibatch = self._pool.get()
#         else:
#             minibatch = self.batch_sampler()

#         if self.batch_idx.empty():
#             # the batch_idx is empty. an "epoch" has passed. resample the
#             # positive index order.
#             print("\treset")
#             # self.reset()

#         return minibatch

#     def _find_positives(self):
#         """Helper function to find the positive samples."""
#         # is it possible to store all the positive examples?
#         # what are the sampling strats:
#         # store all positive examples and train using them.
#         # pick videos and train using negative and positive frames from the
#         # videos.
#         self.pos_locs = numpy.empty((self.num_exp,), dtype=object)
#         pos_idx = []
#         self.num_pos = 0
#         neg_bouts = []

#         for i in range(self.num_exp):
#             exp_name = self.exp_names[i]
#             # print(exp_name)
#             exp = self.data["exps"][exp_name]
#             temp = numpy.argwhere(exp["org_labels"].value > 0)
#             seq_len = exp["org_labels"].shape[0]
#             # print("\t%d" % seq_len)

#             self.pos_locs[i] = temp
#             # self.num_pos += temp.shape[0]

#             # concatenate the index to the experiment name (ie, the key).
#             # Maybe storing the key makes more sense?
#             temp_idx_mat = numpy.concatenate(
#                 [numpy.tile(i, (temp.shape[0], 1)), temp], axis=1
#             )
#             # create the negative bouts
#             prev_idx = 0
#             for j in range(temp_idx_mat.shape[0]):
#                 curr_pos = temp_idx_mat[j][1]

#                 if curr_pos <= 5:
#                     # if the behavior starts at frame 0... just continue
#                     prev_idx = min(curr_pos + 5, seq_len)
#                     continue

#                 bout_len = curr_pos - 5 - prev_idx
#                 # check for negative bout lengths. This can happen when the
#                 # behaviors are too close together.
#                 if bout_len > 0:
#                     bout = numpy.array(
#                         [i, prev_idx, curr_pos - 5, bout_len], ndmin=2)
#                     neg_bouts.append(bout)
#                 prev_idx = min(curr_pos + 5, seq_len)
#                 # if bout_len > 1000 and j != 0:
#                 #     # this seems suspicious...

#                 # # check to make sure that the bout doesn't overlap with
#                 # for k in range(temp_idx_mat.shape[0]):
#                 #     if bout[0, 1] < temp_idx_mat[k][1] and bout[0, 2] > temp_idx_mat[k][1]:

#             # add the last negative bout (goes to the end of the video)
#             if prev_idx != seq_len:
#                 neg_bouts.append(numpy.array(
#                     [i, prev_idx, seq_len, seq_len - prev_idx], ndmin=2))
#             else:
#                 import pdb; pdb.set_trace()
#             pos_idx.append(temp_idx_mat)

#         # concatenate all the pos_idx's into won giant array. First column
#         # is the experiment index, second the frame number for that
#         # experiment and then the third column is the behavior index.
#         self.pos_idx = numpy.concatenate(pos_idx, axis=0)
#         self.num_pos = self.pos_idx.shape[0]

#         self.neg_bouts = numpy.concatenate(neg_bouts, axis=0)
#         self.num_neg = self.neg_bouts[:, 3].sum()

#     def get_frames(self, exp, frame_i):
#         # helper to deal with the frame offsets.
#         imgs = []
#         # num_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#         num_frames = exp["labels"].value.shape[0]

#         frames = []
#         for offset in self.frames:
#             if frame_i + offset < 0:
#                 frame_idx = 0
#             else:
#                 frame_idx = frame_i + offset
#             if frame_i + offset >= num_frames:
#                 frame_idx = num_frames - 1
#             else:
#                 frame_idx = frame_i + offset
#             frames.append(frame_idx)

#         # tic = time.time()
#         raw_imgs = exp["frames"][numpy.array(frames), :]
#         # print(time.time() - tic)
#         for i in range(len(self.frames)):
#             tensor_img = self.preprocess_img(raw_imgs[i])

#             # Need to reshape based on self.frames size. If len(self.frames)
#             # is greater than 1, resize the tensor_img so it can be cat'ed
#             # with the rest of the frames.
#             tensor_shape = tensor_img.size()
#             if len(self.frames) > 1:
#                 tensor_img.resize_(
#                     tensor_shape[0], 1, *tensor_shape[1:])

#             imgs.append(tensor_img)

#         # if the number of frames is 1, then just take the first element.
#         # otherwise, cat on the 2nd dimension.
#         if len(self.frames) == 1:
#             feat = imgs[0]
#         else:
#             feat = torch.cat(imgs, 1)

#         return feat

#     def get_batch(self, batch_size, idx, idx_range):
#         # get the batch
#         # first get the batch features and labels
#         if len(self.frames) == 1:
#             temp_feat = torch.zeros(batch_size, 3, 224, 224)
#         else:
#             temp_feat = torch.zeros(batch_size, 3, len(self.frames), 224, 224)

#         labels = torch.zeros(batch_size, self.label_dims)

#         for sample_i in range(batch_size):
#             exp_i = idx[idx_range[sample_i]][0]
#             frame_i = idx[idx_range[sample_i]][1]
#             exp_name = self.exp_names[exp_i]
#             # video_name = self.data["exps"][exp_name]["video_name"].value
#             exp = self.data["exps"][exp_name]

#             # get the desired frames.
#             # cap = cv2.VideoCapture(
#             #     os.path.join(self.video_path, video_name))
#             # import pdb; pdb.set_trace()
#             temp_feat[sample_i, :] = self.get_frames(exp, frame_i)
#             # cap.release()

#             label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
#             labels[sample_i, :] = torch.from_numpy(label)

#         return temp_feat, labels

#     def batch_sampler(self):
#         batch_id = self.batch_idx.get()
#         # print "worker %d: loaded data, waiting to put..." % self.threadid
#         # now actually get the batch
#         # print("a")
#         idx_range = list(range(
#             batch_id * self._pos_batch_size,
#             (batch_id + 1) * self._pos_batch_size))

#         # Loop over the pos_idx's. This array is shuffled at each epoch.
#         # print(self.pos_idx[idx_range, :])
#         # print("b")
#         pos_features, pos_labels =\
#             self.get_batch(self._pos_batch_size, self.pos_idx, idx_range)

#         # figure out the negative samples.
#         idx_range = list(range(
#             batch_id * self._neg_batch_size,
#             (batch_id + 1) * self._neg_batch_size))
#         # print(idx_range)
#         # neg_features = []
#         neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)
#         # to use the get_batch function, merge the frame and exp info, and
#         # create a new idx_range. get_batch was mostly designed to grab frames
#         # from the pos_idx, which is a list of all positives in the video.
#         neg_exp_frame = numpy.asarray([neg_exps_idx, neg_frames_idx]).T
#         # print(neg_exp_frame)
#         neg_features, neg_labels = self.get_batch(
#             self._neg_batch_size, neg_exp_frame, range(self._neg_batch_size))
#         # print("c")
#         # check to see if negative samples are too close to positive samples.
#         if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
#             import pdb; pdb.set_trace()

#         # concatenate the positive and negative examples
#         # Potential speed upgrade:
#         # https://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
#         # features = [
#         #     numpy.concatenate([pos, neg], axis=0)
#         #     for pos, neg in zip(pos_features, neg_features)
#         # ]
#         # label = numpy.concatenate([pos_labels, neg_labels], axis=0)
#         # features = [
#         #     torch.cat([pos, neg], dim=0)
#         #     for pos, neg in zip(pos_features, neg_features)
#         # ]
#         # import pdb; pdb.set_trace()
#         features = [
#             torch.cat([pos_features, neg_features], dim=0)
#         ]
#         labels = torch.cat([pos_labels, neg_labels])
#         # inputs = features + [labels]
#         # inputs = [inputs[0][-1:, :], inputs[1][-1:, :], inputs[2][-1:]]
#         # labels = self._convert_labels(labels)

#         # feat_var = [
#         #     Variable(var, requires_grad=True) for var in features
#         # ]
#         # label_var = [Variable(labels, requires_grad=False)]
#         feat_var = features
#         label_var = [labels]

#         inputs = feat_var + label_var
#         if self.use_gpu >= 0:
#             inputs = [
#                 input.cuda(self.use_gpu) for input in inputs
#             ]
#         # print("d")
#         return inputs

#     def _find_neg_exp_frame(self, idx_range):
#         """Helper function to find the negative samples."""
#         # this is a weird way to do this... maybe doesn't make sense. This
#         # enumerates all negative examples and searches each bout for the
#         # deisred sample. So if we are looking for negative sample 10, and
#         # the negative bouts are of length 2, 4, 8. Then we'd take the 4th
#         # sample in the 3rd negative bout (of length 8), ie the 10th negative
#         # sample of all the samples.
#         # to improve the speed of this, sort the batch indexes. Then we can
#         # search the bouts in one pass for each sample in the batch
#         samples_idx = self.neg_idx[idx_range]
#         samples_idx.sort()

#         # row_i = 0
#         # seen_frames = 0
#         # remainder = idx
#         # prev_bout = 0
#         remainder = 0

#         exps_idx = []
#         frames_idx = []

#         bout_cumsum = numpy.cumsum(self.neg_bouts[:, 3])
#         for sample_i in samples_idx:
#             # search for the bouts for the frame/exp info. Use the
#             # numpy.cumsum function to figure out which bout contains the
#             # nth negative sample
#             try:
#                 idx = numpy.argwhere(bout_cumsum > sample_i)[0][0]
#             except:
#                 print("Bad negative sample, greater than max number of negative frames")
#                 exit()
#             exps_idx.append(self.neg_bouts[idx, 0])
#             if idx == 0:
#                 num_seen_frames = 0
#             else:
#                 num_seen_frames = bout_cumsum[idx - 1]
#             remainder = sample_i - num_seen_frames
#             frames_idx.append(self.neg_bouts[idx, 1] + remainder)

#         # re-shuffle the samples? probably not needed in most cases... but
#         # this would be a good place to do it.
#         return exps_idx, frames_idx

#     def check_negatives(self, neg_exps_idx, neg_frames_idx):
#         """Make sure no negative frame is too close to a positive."""
#         for neg_i in range(self._neg_batch_size):
#             neg_exp = neg_exps_idx[neg_i]
#             neg_frame = neg_frames_idx[neg_i]
#             for i in range(self.num_pos):
#                 if self.pos_idx[i, 0] == neg_exp:
#                     pos_frame = self.pos_idx[i, 1]
#                     if numpy.abs(pos_frame - neg_frame) < 5:
#                         return True
#         return False

#     def preprocess_img(self, img):
#         """Apply pytorch preprocessing."""
#         # try:
#         img = cv2.resize(img, (self.width, self.height))
#         # except:
#         # import pdb; pdb.set_trace()
#         pil_img = PIL.Image.fromarray(img)
#         tensor_img = self.preproc(pil_img)

#         return tensor_img
