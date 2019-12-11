import numpy
from helpers import DataLoader
# import torch
# import torchvision.transforms as transforms
import PIL
import os
import cv2
# python 2 vs 3 stuff...
import sys
import gflags
if sys.version_info[0] < 3:
    import Queue as queue
else:
    import queue
# import time
# from torch.autograd import Variable

gflags.DEFINE_boolean("use_pool", False, "Use a pool (threads) for the sampler")
gflags.DEFINE_integer("max_workers", 2, "Max number of workers.")
gflags.DEFINE_integer("max_queue", 5, "Maximum queue length.")


class HDF5Sampler(object):
    """HDF5Sampler

    Samples HDF5 files. Retrives the provided keys.
    """
    def __init__(self, rng, hdf5_data, mini_batch, feat_keys,
                 max_workers=2, seq_len=-1, max_queue=5,
                 use_pool=False, gpu_id=-1, feat_pre=None, label_pre=None):
        self.rng = rng
        self.data = hdf5_data
        self.use_pool = use_pool
        self.use_gpu = gpu_id
        self.exp_names = self.data["exp_names"][()]
        self.label_names = self.data["label_names"][()]
        self.seq_len = seq_len

        self.feat_keys = feat_keys
        # get the dimensions of each requested feature. Check the
        # first experiment's feature dimensions.
        self.feat_dims = [
            self._get_field(self.exp_names[0], feat_key).shape[1]
            for feat_key in self.feat_keys
        ]

        self.label_dims =\
            self.data["exps"][self.exp_names[0]]["labels"].shape[1]

        self.mini_batch = mini_batch
        self.num_exp = len(self.exp_names)

        self.num_batch = int(self.num_exp / self.mini_batch)
        self.batch_idx = queue.Queue(self.num_batch)
        self.exp_idx = []  # this is set in reset

        self.feat_pre = feat_pre
        self.label_pre = label_pre

        if self.use_pool is True:
            self._pool = DataLoader.DataLoaderPool(
                sampler=self._batch_sampler, max_workers=max_workers,
                max_queue=max_queue
            )

    def reset(self):
        """Reset the sampler."""
        self.exp_idx = numpy.asarray(range(len(self.exp_names)))
        if self.rng is not None:
            self.rng.shuffle(self.exp_idx)
        for i in range(self.num_batch):
            self.batch_idx.put(i)

    def _get_field(self, exp_name, feat_key_list):
        """Get the feature from the hdf file.

        The feature key list could be arbitrarily long. To help deal
        with HDF files that have more structure, this function will
        traverse the HDF dictionaries using the keys in feat_key_list.
        """
        # this might make more sense as a recursive function, but
        # traversing the list as a for loop seems to be just as
        # easy.
        curr_field = self.data["exps"][exp_name]

        for feat_key in feat_key_list:
            curr_field = curr_field[feat_key]

        return curr_field[()]

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self._batch_sampler()

        return minibatch

    def _batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id * self.mini_batch,
            (batch_id + 1) * self.mini_batch))

        # # preallocate the space
        # features = torch.zeros(
        #     self.seq_len, self.channels, self.width, self.height)
        # labels = torch.zeros(self.seq_len, self.label_dims)

        # get the exps
        sampled_idx = self.exp_idx[idx_range]
        sampled_exps = self.exp_names[sampled_idx]

        # get the features
        features, labels, proc_labels, masks, exps =\
            self._get_features(sampled_exps)

        # if self.use_gpu >=0:
        #     # put the data onto the gpu.
        #     features = [
        #         feature.cuda() for feature in features
        #     ]
        #     labels = labels.cuda()
        # import pdb; pdb.set_trace()
        # feat_var = [
        #     Variable(var, requires_grad=True) for var in features
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        # feat_var = [features]
        # label_var = [labels]

        features = [
            feature
        ]
        labels = labels
        masks = masks
        proc_labels = proc_labels

        # if self.use_gpu >= 0:
        #     # put features on the gpu
        #     features = [
        #         features.cuda() for features in features
        #     ]
        #     labels = labels.cuda()
        #     masks = masks.cuda()
        #     proc_labels = proc_labels.cuda()
        inputs = {
            "features": features,
            "labels": labels,
            "proc_labels": proc_labels,
            "masks": masks,
            "names": sampled_exps
        }
        return inputs

    def _get_features(self, sampled_exps):
        """Get the features for the sampled_exps."""
        # sequence dimensions are assumed to be:
        # seq_len x batch x features (or labels)
        all_feats = [
            numpy.zeros(
                (self.seq_len, self.mini_batch, feat_dim), dtype="float32")
            for feat_dim in self.feat_dims
        ]
        all_labels = numpy.zeros(
            (self.seq_len, self.mini_batch, self.label_dims),
            dtype="float32")
        all_proc_labels = numpy.zeros(
            (self.seq_len, self.mini_batch, self.label_dims),
            dtype="float32")
        all_masks = numpy.zeros(
            (self.seq_len, self.mini_batch, self.label_dims),
            dtype="float32")
        for i in range(len(sampled_exps)):
            # len(sampled_exps) should be the same as self.mini_batch
            exp_name = sampled_exps[i]
            curr_label = self.data["exps"][exp_name]["labels"][()]
            seq_len = numpy.min([curr_label.shape[0], self.seq_len])

            all_labels[:seq_len, i, :] = curr_label[:seq_len, :]
            all_masks[:seq_len, i, :] = 1

            for j in range(len(self.feat_keys)):
                feat = self._get_field(sampled_exps[i],
                                       self.feat_keys[j])
                if self.feat_pre is not None:
                    feat[:seq_len, :] = self.feat_pre(
                        feat[:seq_len, :], self.feat_keys[j])
                all_feats[j][:seq_len, i, :] = feat[:seq_len, :]

            if self.label_pre is not None:
                all_proc_labels[:seq_len, i, :] = self.label_pre(all_labels[:seq_len, i, :])
            else:
                all_proc_labels[:seq_len, i, :] = all_labels[:seq_len, i, :]

        return all_feats, all_labels, all_proc_labels, all_masks, sampled_exps


class VideoFrameSampler(object):
    """VideoFrameSampler

    Samples random frames from videos.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, video_path, mini_batch, max_workers=2,
                 frames=[0], max_queue=5, use_pool=False, gpu_id=-1,
                 normalize=None, channels=3, width=224, height=224):
        self.rng = rng
        self.data = hdf_data
        self.video_path = video_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id
        self.frames = frames
        self.exp_names = self.data["exp_names"]
        self.label_dims =\
            self.data["exps"][self.exp_names[0]]["labels"].shape[1]
        self.channels = channels
        self.width = width
        self.height = height

        self.mini_batch = mini_batch
        # Because positives are so rare, we'll want to actively find positive
        # samples for the minibatch. This variable will deterimine how many
        # positive samples are needed per patch.
        self._pos_batch_size = int(numpy.ceil(mini_batch * 0.5))
        self._neg_batch_size = mini_batch - self._pos_batch_size

        # self.num_exp = 3
        self.num_exp = len(self.exp_names)
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].shape[1]

        self.pos_batch_idx = 0
        # the following values will be setup by _find_positives()
        self.pos_locs = []
        self.num_pos = 0
        self.num_neg = 0
        # First column is the experiment index, second the frame number for
        # that experiment and then the third column is the behavior index.
        self.pos_idx = []
        # Each index is a scalar that is between 0 and number of negative
        # samples. The scalar represents a frame that indexes into the
        # self.neg_bouts variable.
        # For example, if self.neg_bouts[0] = [0, 0, 120, 120] and
        # self.neg_bouts[1] = [0, 130, 146, 16], and self.neg_idx[0] = 122.
        # Then self.neg_idx[i] represents frame 132 of experiment 0.
        self.neg_idx = []
        # Find the positive examples (and figure out the negative bouts).
        self._find_positives()

        # compute the number of "batches" that can be created with the number
        # of positive examples.
        # Ignore remainders? Assume leftovers will be seen by the network
        # eventually.
        self.num_batch = int(self.num_pos / self._pos_batch_size)
        self.batch_idx = queue.Queue(self.num_batch)
        # setup the queue.
        # self.reset()

        # setup the preprocessing
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        if normalize is None:
            self.preproc = lambda x: x
        else:
            self.preproc = normalize
        # if normalize is None:
        #     self.preproc = transforms.Compose([
        #         transforms.Resize([224, 224]),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor()
        #         # normalize
        #     ])
        # else:
        #     self.preproc = transforms.Compose([
        #         transforms.Resize([224, 224]),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize
        #     ])

        # initialize the workers if necessary.
        if self.use_pool is True:
            self._pool = DataLoader.DataLoaderPool(
               sampler=self.batch_sampler, max_workers=max_workers,
               max_queue=max_queue
            )

    def reset(self):
        """Reset the sampler."""
        self.rng.shuffle(self.pos_idx)
        for i in range(self.num_batch):
            self.batch_idx.put(i)

        # create a random sequence of negative
        self.neg_idx = self.rng.choice(
            self.num_neg,
            (self._neg_batch_size * self.num_batch),
            replace=False)
        # self.half_batch_idx = 0
        # print("\treset!")

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        # if self.batch_idx.empty():
        #     # the batch_idx is empty. an "epoch" has passed. resample the
        #     # positive index order.
        #     print("\treset")
        #     # self.reset()

        return minibatch

    def _find_positives(self):
        """Helper function to find the positive samples."""
        # is it possible to store all the positive examples?
        # what are the sampling strats:
        # store all positive examples and train using them.
        # pick videos and train using negative and positive frames from the
        # videos.
        self.pos_locs = numpy.empty((self.num_exp,), dtype=object)
        pos_idx = []
        self.num_pos = 0
        neg_bouts = []

        for i in range(self.num_exp):
            exp_name = self.exp_names[i]
            # print(exp_name)
            exp = self.data["exps"][exp_name]
            temp = numpy.argwhere(exp["labels"][()] > 0)
            seq_len = exp["labels"].shape[0]
            # print("\t%d" % seq_len)

            self.pos_locs[i] = temp
            # self.num_pos += temp.shape[0]

            # concatenate the index to the experiment name (ie, the key).
            # Maybe storing the key makes more sense?
            temp_idx_mat = numpy.concatenate(
                [numpy.tile(i, (temp.shape[0], 1)), temp], axis=1
            )
            # create the negative bouts
            prev_idx = 0
            for j in range(temp_idx_mat.shape[0]):
                curr_pos = temp_idx_mat[j][1]

                if curr_pos <= 5:
                    # if the behavior starts at frame 0... just continue
                    prev_idx = min(curr_pos + 5, seq_len)
                    continue

                bout_len = curr_pos - 5 - prev_idx
                # check for negative bout lengths. This can happen when the
                # behaviors are too close together.
                if bout_len > 0:
                    bout = numpy.array(
                        [i, prev_idx, curr_pos - 5, bout_len], ndmin=2)
                    neg_bouts.append(bout)
                prev_idx = min(curr_pos + 5, seq_len)
                # if bout_len > 1000 and j != 0:
                #     # this seems suspicious...

                # # check to make sure that the bout doesn't overlap with
                # for k in range(temp_idx_mat.shape[0]):
                #     if bout[0, 1] < temp_idx_mat[k][1] and bout[0, 2] > temp_idx_mat[k][1]:

            # add the last negative bout (goes to the end of the video)
            if prev_idx != seq_len:
                neg_bouts.append(numpy.array(
                    [i, prev_idx, seq_len, seq_len - prev_idx], ndmin=2))
            # else:
            #     import pdb; pdb.set_trace()
            pos_idx.append(temp_idx_mat)

        # concatenate all the pos_idx's into won giant array. First column
        # is the experiment index, second the frame number for that
        # experiment and then the third column is the behavior index.
        self.pos_idx = numpy.concatenate(pos_idx, axis=0)
        self.num_pos = self.pos_idx.shape[0]

        self.neg_bouts = numpy.concatenate(neg_bouts, axis=0)
        self.num_neg = self.neg_bouts[:, 3].sum()

    def get_frames(self, cap, frame_i):
        # helper to deal with the frame offsets.
        imgs = []
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for offset in self.frames:
            if frame_i + offset < 0:
                frame_idx = 0
            else:
                frame_idx = frame_i + offset
            if frame_i + offset >= num_frames:
                frame_idx = num_frames - 1
            else:
                frame_idx = frame_i + offset

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            retval, img = cap.read()

            tensor_img = self.preprocess_img(img)
            # Need to reshape based on self.frames size. If len(self.frames)
            # is greater than 1, resize the tensor_img so it can be cat'ed
            # with the rest of the frames.
            tensor_shape = tensor_img.shape
            if len(self.frames) > 1:
                tensor_img = tensor_img.reshape(
                    (1, *tensor_shape))

            imgs.append(tensor_img)

        # if the number of frames is 1, then just take the first element.
        # otherwise, cat on the 2nd dimension.
        if len(self.frames) == 1:
            feat = imgs[0]
        else:
            feat = numpy.concatenate(imgs, axis=0)

        return feat

    def get_batch(self, batch_size, idx, idx_range):
        # get the batch
        # first get the batch features and labels
        # if len(self.frames) == 1:
        #     temp_feat = torch.zeros(batch_size, 3, 224, 224)
        # else:
        #     temp_feat = torch.zeros(batch_size, 3, len(self.frames), 224, 224)
        if len(self.frames) == 1:
            # temp_feat = numpy.zeros((batch_size, 3, 224, 224)).astype("float32")
            temp_feat = numpy.zeros((batch_size, 224, 224, 3)).astype("float32")
        else:
            # temp_feat = numpy.zeros(
            #     (batch_size, 3, len(self.frames), 224, 224)).astype("float32")
            temp_feat = numpy.zeros(
                (batch_size, len(self.frames), 224, 224, 3)).astype("float32")

        # labels = torch.zeros(batch_size, self.label_dims)
        exp_names = []
        frame_nums = []
        exp_idx = []
        label_idx = []
        labels = numpy.zeros((batch_size, self.label_dims))
        for sample_i in range(batch_size):
            exp_i = idx[idx_range[sample_i]][0]
            frame_i = idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]
            # if "video_name" not in self.data["exps"][exp_name].keys():
            #     video_name = "%s.avi" % exp_name
            #     video_name = video_name.decode()
            # else:
            #     video_name = self.data["exps"][exp_name]["video_name"][()]
            video_name = "%s.avi" % (exp_name.decode())
            # video_name = "%s.mp4" % (exp_name.decode())

            # get the desired frames.
            # print(os.path.join(self.video_path, video_name))
            cap = cv2.VideoCapture(
                os.path.join(self.video_path, video_name))
            temp_feat[sample_i, :] = self.get_frames(cap, frame_i)
            cap.release()
            # label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
            label = self.data["exps"][exp_name]["labels"][frame_i, :]
            # labels[sample_i, :] = torch.from_numpy(label)
            labels[sample_i, :] = label

            # exp_names.append(exp_name.decode())
            # frame_nums.append(frame_i)
            # exp_idx.append(exp_i)
            # label_idx.append(idx[idx_range[sample_i]][2])

        return temp_feat, labels# , exp_names, frame_nums, label_idx

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        # print("a")
        idx_range = list(range(
            batch_id * self._pos_batch_size,
            (batch_id + 1) * self._pos_batch_size))

        # Loop over the pos_idx's. This array is shuffled at each epoch.
        # print(self.pos_idx[idx_range, :])
        # print("b")
        # pos_features, pos_labels, pos_names, pos_frames, pos_labels =\
        pos_features, pos_labels =\
          self.get_batch(self._pos_batch_size, self.pos_idx, idx_range)

        # figure out the negative samples.
        idx_range = list(range(
            batch_id * self._neg_batch_size,
            (batch_id + 1) * self._neg_batch_size))
        # print(idx_range)
        # neg_features = []
        neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)
        # to use the get_batch function, merge the frame and exp info, and
        # create a new idx_range. get_batch was mostly designed to grab frames
        # from the pos_idx, which is a list of all positives in the video.
        neg_exp_frame = numpy.asarray([neg_exps_idx, neg_frames_idx]).T

        # neg_features, neg_labels, neg_names, neg_frames, neg_labels =\
        neg_features, neg_labels =\
          self.get_batch(
            self._neg_batch_size, neg_exp_frame, range(self._neg_batch_size))
        # print("c")
        # check to see if negative samples are too close to positive samples.
        if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
            import pdb; pdb.set_trace()

        # features = [
        #     torch.cat([pos_features, neg_features], dim=0)
        # ]
        # labels = torch.cat([pos_labels, neg_labels])
        features = [
            numpy.concatenate([pos_features, neg_features], axis=0)
        ]
        labels = numpy.concatenate([pos_labels, neg_labels], axis=0)
        # exp_names = numpy.concatenate([pos_names, neg_names])
        # frame_idx = numpy.concatenate([pos_frames, neg_frames])
        # label_idx = numpy.concatenate([pos_labels, neg_labels])

        feat_var = features
        label_var = [labels]

        inputs = feat_var + label_var
        # inputs = feat_var + label_var + [{
        #     'features': features,
        #     'labels': labels,
        #     'frame_idx': frame_idx,
        #     'label_idx': label_idx
        # }]
        # if self.use_gpu >= 0:
        #     inputs = [
        #         input.cuda(self.use_gpu) for input in inputs
        #     ]
        # print("d")
        return inputs

    def _find_neg_exp_frame(self, idx_range):
        """Helper function to find the negative samples."""
        # this is a weird way to do this... maybe doesn't make sense. This
        # enumerates all negative examples and searches each bout for the
        # deisred sample. So if we are looking for negative sample 10, and
        # the negative bouts are of length 2, 4, 8. Then we'd take the 4th
        # sample in the 3rd negative bout (of length 8), ie the 10th negative
        # sample of all the samples.
        # to improve the speed of this, sort the batch indexes. Then we can
        # search the bouts in one pass for each sample in the batch
        samples_idx = self.neg_idx[idx_range]
        samples_idx.sort()

        # row_i = 0
        # seen_frames = 0
        # remainder = idx
        # prev_bout = 0
        remainder = 0

        exps_idx = []
        frames_idx = []

        bout_cumsum = numpy.cumsum(self.neg_bouts[:, 3])
        for sample_i in samples_idx:
            # search for the bouts for the frame/exp info. Use the
            # numpy.cumsum function to figure out which bout contains the
            # nth negative sample
            try:
                idx = numpy.argwhere(bout_cumsum > sample_i)[0][0]
            except:
                print("Bad negative sample, greater than max number of negative frames")
                exit()
            exps_idx.append(self.neg_bouts[idx, 0])
            if idx == 0:
                num_seen_frames = 0
            else:
                num_seen_frames = bout_cumsum[idx - 1]
            remainder = sample_i - num_seen_frames
            frames_idx.append(self.neg_bouts[idx, 1] + remainder)

        # re-shuffle the samples? probably not needed in most cases... but
        # this would be a good place to do it.
        return exps_idx, frames_idx

    def check_negatives(self, neg_exps_idx, neg_frames_idx):
        """Make sure no negative frame is too close to a positive."""
        for neg_i in range(self._neg_batch_size):
            neg_exp = neg_exps_idx[neg_i]
            neg_frame = neg_frames_idx[neg_i]
            for i in range(self.num_pos):
                if self.pos_idx[i, 0] == neg_exp:
                    pos_frame = self.pos_idx[i, 1]
                    if numpy.abs(pos_frame - neg_frame) < 5:
                        return True
        return False

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # try:
        img = cv2.resize(img, (self.width, self.height))
        # except:
        # pil_img = PIL.Image.fromarray(img)
        tensor_img = self.preproc(img)

        return tensor_img


class VideoSampler(object):
    """VideoSampler

    Samples random videos.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, video_path, max_workers=2,
                 seq_len=1500, max_queue=5, use_pool=False,
                 gpu_id=-1, normalize=None, channels=3, width=224, height=224):
        self.rng = rng
        self.data = hdf_data
        self.video_path = video_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id
        self.exp_names = self.data["exp_names"][()]
        self.exp_names.sort()
        self.label_dims =\
            self.data["exps"][self.exp_names[0]]["labels"].shape[1]
        self.channels = channels
        self.width = width
        self.height = height
        self.seq_len = seq_len

        self.num_exp = len(self.exp_names)
        # self.num_exp = 4
        self.exp_ids = numpy.asarray(range(self.num_exp))
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].shape[1]

        # compute the number of "batches" that can be created with the number
        # of positive examples.
        # Ignore remainders? Assume leftovers will be seen by the network
        # eventually.
        self.num_batch = self.num_exp
        self.batch_idx = queue.Queue(self.num_batch)
        # setup the queue.
        # self.reset()

        # setup the preprocessing
        if normalize is None:
            # Scale doesn't seem to be taking in a sequence, even tho the
            # help says it should...
            self.preproc = transforms.Compose([
                transforms.Resize([self.width, self.height]),
                # transforms.Scale([self.width, self.height]),
                # transforms.CenterCrop(224),
                transforms.ToTensor()
                # normalize
            ])
        else:
            self.preproc = transforms.Compose([
                transforms.Resize([self.width, self.height]),
                # transforms.Scale([self.width, self.height]),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        # initialize the workers if necessary.
        if self.use_pool is True:
            self._pool = DataLoader.DataLoaderPool(
               sampler=self.batch_sampler, max_workers=max_workers,
               max_queue=max_queue
            )

    def reset(self):
        """Reset the sampler."""
        if self.rng is not None:
            self.rng.shuffle(self.exp_ids)
        for i in range(self.num_batch):
            self.batch_idx.put(self.exp_ids[i])
        # self.half_batch_idx = 0
        # print("\treset!")

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        # if self.batch_idx.empty():
        #     # the batch_idx is empty. an "epoch" has passed. resample the
        #     # positive index order.
        #     print("\treset")
        #     # self.reset()

        return minibatch

    def get_frames(self, exp):
        # helper to deal with the frame offsets.
        exp_name = self.exp_names[exp]
        video_name = self.data["exps"][exp_name]["video_name"][()]
        # print(exp_name)

        cap = cv2.VideoCapture(
            os.path.join(self.video_path, video_name))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # pre-allocate the space
        np_labels = self.data["exps"][exp_name]["labels"][()]
        if self.seq_len > -1:
            # frames = torch.zeros(
            #     self.seq_len, self.channels, self.width, self.height)
            # labels = torch.zeros(self.seq_len, self.label_dims)
            # num_frames = numpy.min([num_frames, self.seq_len])
            frames = torch.numpy(
                (self.seq_len, self.channels, self.width,
                 self.height)).astype("float32")
            labels = torch.numpy(
                (self.seq_len, self.label_dims)).astype("float32")
            num_frames = numpy.min([num_frames, self.seq_len])
        else:
            # frames = torch.zeros(
            #     num_frames, self.channels, self.width, self.height)
            # labels = torch.zeros(num_frames, self.label_dims)
            frames = numpy.zeros(
                (num_frames, self.channels, self.width,
                 self.height)).astype("float32")
            labels = numpy.zeros(
                (num_frames, self.label_dims)).astype("float32")
        # the number of frames to process is the minimum of the desired
        # sequence length and the number of frames in the video.
        # tic = time.time()

        for frame_i in range(num_frames):
            # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_i)
            retval, img = cap.read()
            # cv2.imshow("img", img)
            # cv2.waitKey(int(1.0 / 30.0 * 1000))
            tensor_img = self.preprocess_img(img)
            frames[frame_i, :] = tensor_img

        # get the labels
        num_labels = np_labels.shape[0]
        num_labels = numpy.min([num_frames, num_labels])
        # labels = torch.zeros(num_frames, self.label_dims)
        # import pdb; pdb.set_trace()
        # labels[:num_labels, :] = torch.from_numpy(np_labels)[:num_labels, :]
        labels[:num_labels, :] = np_labels[:num_labels, :]

        cap.release()

        return frames, labels

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id,
            batch_id + 1))

        # # preallocate the space
        # features = torch.zeros(
        #     self.seq_len, self.channels, self.width, self.height)
        # labels = torch.zeros(self.seq_len, self.label_dims)

        # get the exps
        sampled_exps = self.exp_ids[idx_range]

        cur_feats, cur_labels =\
            self.get_frames(sampled_exps[0])
        features = cur_feats
        labels = cur_labels
        # import pdb; pdb.set_trace()
        # feat_var = [
        #     Variable(var, requires_grad=True) for var in features
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        feat_var = [features]
        label_var = [labels]

        inputs = feat_var + label_var
        if self.use_gpu >= 0:
            inputs = [
                input.cuda(self.use_gpu) for input in inputs
            ]
        inputs = inputs + [self.exp_names[sampled_exps]]
        return inputs

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # split the image into two frames
        img = cv2.resize(img, (self.width, self.height))
        # pil_img = PIL.Image.fromarray(img)
        tensor_img = self.preproc(img)

        return tensor_img


class HantmanVideoSampler(VideoSampler):
    def __init__(self, rng, hdf_data, video_path, max_workers=2,
                 seq_len=1500, max_queue=5, use_pool=False,
                 gpu_id=-1, channels=3, width=224, height=224):
        # setup the preprocessing, default normalization...
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        super(HantmanVideoSampler, self).__init__(
            rng, hdf_data, video_path, max_workers=max_workers,
            seq_len=seq_len, max_queue=max_queue, use_pool=use_pool,
            gpu_id=gpu_id, normalize=normalize,
            channels=channels, width=width, height=height)

    def get_frames(self, exp):
        # helper to deal with the frame offsets.
        # differnce from super class in the video name as well as the output
        # frame size.
        exp_name = self.exp_names[exp].decode("utf-8")
        video_name = os.path.join(self.video_path, exp_name, "movie_comb.avi")
        # print(exp_name)

        cap = cv2.VideoCapture(video_name)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        half_width = (int)(width / 2)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        np_labels = self.data["exps"][exp_name]["labels"][()]
        # num_labels = numpy.min(
        #     [num_frames, np_labels.shape[0]])

        # # get the labels
        # if self.seq_len > -1:
        #     frames1 = torch.zeros(
        #         self.seq_len, self.channels, self.width, self.height)
        #     frames2 = torch.zeros(
        #         self.seq_len, self.channels, self.width, self.height)
        #     labels = torch.zeros(self.seq_len, self.label_dims)

        #     num_frames = numpy.min([num_frames, self.seq_len])
        #     num_labels = numpy.min([num_labels, num_frames])
        # else:
        #     num_frames = numpy.min([num_frames, num_labels])
        #     frames1 = torch.zeros(
        #         num_frames, self.channels, self.width, self.height)
        #     frames2 = torch.zeros(
        #         num_frames, self.channels, self.width, self.height)
        #     labels = torch.zeros(num_frames, self.label_dims)

        #     num_labels = numpy.min([num_frames, num_labels])

        # The number of frames/labels is the minimum of seq_len, video
        # frames, and number of labels. Kind of a speed hack.
        num_frames = numpy.min(
            [num_frames, np_labels.shape[0], self.seq_len]
        )

        # frames1 = torch.zeros(
        #     num_frames, self.channels, self.width, self.height)
        # frames2 = torch.zeros(
        #     num_frames, self.channels, self.width, self.height)
        # labels = torch.zeros(num_frames, self.label_dims)
        frames1 = numpy.zeros(
            (num_frames, self.channels, self.width,
             self.height)).astype("float32")
        frames2 = numpy.zeros(
            (num_frames, self.channels, self.width,
             self.height)).astype("float32")
        labels = numpy.zeros((num_frames, self.label_dims)).astype("float32")

        # labels[:num_frames, :] = torch.from_numpy(np_labels)[:num_frames]
        labels[:num_frames, :] = np_labels[:num_frames]

        # the number of frames to process is the minimum of the desired
        # sequence length and the number of frames in the video.
        # tic = time.time()
        # pre-allocate the space
        for frame_i in range(num_frames):
            # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_i)
            retval, img = cap.read()
            # cv2.imshow("img", img)
            # cv2.waitKey(int(1.0 / 30.0 * 1000))
            img1 = img[:, :half_width, :]
            img2 = img[:, half_width:, :]
            # import pdb; pdb.set_trace()
            tensor_img1 = self.preprocess_img(img1)
            frames1[frame_i, 0, :] = tensor_img1[0, :]
            tensor_img2 = self.preprocess_img(img2)
            frames2[frame_i, 0, :] = tensor_img2[0, :]

        # print(time.time() - tic)

        cap.release()

        return [frames1, frames2], labels

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id,
            (batch_id + 1)))

        # # preallocate the space
        # features1 = torch.zeros(
        #     self.seq_len, self.channels, self.width, self.height)
        # features2 = torch.zeros(
        #     self.seq_len, self.channels, self.width, self.height)
        # org_labels = torch.zeros(self.seq_len, self.label_dims)
        # labels = torch.zeros(self.seq_len, 1, self.label_dims)

        # get the exps
        sampled_exps = self.exp_ids[idx_range]
        # print(sampled_exps)
        # import pdb; pdb.set_trace()
        # for sample_i in range(len(sampled_exps)):
        cur_feats, cur_labels =\
            self.get_frames(sampled_exps[0])
        features1 = cur_feats[0]
        features2 = cur_feats[1]
        labels = cur_labels

        # feat_var = [
        #     Variable(var, requires_grad=True) for var in features
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        feat_var = [features1, features2]
        label_var = [labels]

        inputs = feat_var + label_var
        # if self.use_gpu >= 0:
        #     inputs = [
        #         input.cuda(self.use_gpu) for input in inputs
        #     ]
        inputs = inputs + [self.exp_names[sampled_exps]]
        return inputs


class HantmanVideoFrameSampler(VideoFrameSampler):
    """HantmanVideoFrameSampler

    Samples random frames from videos.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, video_path, mini_batch, max_workers=2,
                 frames=[0], max_queue=5, use_pool=False, gpu_id=-1,
                 normalize=None, channels=3, width=224, height=224):
        # setup the preprocessing, default normalization...
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        super(HantmanVideoFrameSampler, self).__init__(
            rng, hdf_data, video_path, mini_batch, max_workers=max_workers,
            frames=frames, max_queue=max_queue, use_pool=use_pool,
            gpu_id=gpu_id, normalize=normalize,
            channels=channels, width=width, height=height)

    def get_frames(self, cap, frame_i):
        # helper to deal with the frame offsets.
        preproc_imgs1 = []
        preproc_imgs2 = []
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # just in case the hantman videos size change...
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        half_width = (int)(width / 2)
        for offset in self.frames:
            if frame_i + offset < 0:
                frame_idx = 0
            else:
                frame_idx = frame_i + offset
            if frame_i + offset >= num_frames:
                frame_idx = num_frames - 1
            else:
                frame_idx = frame_i + offset

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            retval, img = cap.read()

            img1 = img[:, :half_width, :]
            img2 = img[:, half_width:, :]

            tensor_img1 = self.preprocess_img(img1)
            tensor_img2 = self.preprocess_img(img2)
            # no need to make it black and white
            # tensor_img1 = tensor_img1[:1, :, :]
            # tensor_img2 = tensor_img2[:1, :, :]

            # Need to reshape based on self.frames size. If len(self.frames)
            # is greater than 1, resize the tensor_img so it can be cat'ed
            # with the rest of the frames.
            # tensor_shape = tensor_img1.size()
            tensor_shape = tensor_img1.shape
            if len(self.frames) > 1:
                tensor_img1 = tensor_img1.reshape(
                    (1, *tensor_shape))
                tensor_img2 = tensor_img2.reshape(
                    (1, *tensor_shape))

            preproc_imgs1.append(tensor_img1)
            preproc_imgs2.append(tensor_img2)
        # if the number of frames is 1, then just take the first element.
        # otherwise, cat on the 2nd dimension.
        if len(self.frames) == 1:
            feat = [preproc_imgs1[0], preproc_imgs2[0]]
        else:
            # feat = [
            #     torch.cat(preproc_imgs1, 1),
            #     torch.cat(preproc_imgs2, 1)
            # ]
            feat = [
                numpy.concatenate(preproc_imgs1, axis=0),
                numpy.concatenate(preproc_imgs2, axis=0)
            ]

        return feat

    def get_batch(self, batch_size, idx, idx_range):
        # get the batch
        # first get the batch features and labels
        # if len(self.frames) == 1:
        #     frames1 = torch.zeros(batch_size, 1, 224, 224)
        #     frames2 = torch.zeros(batch_size, 1, 224, 224)
        # else:
        #     frames1 = torch.zeros(batch_size, 1, len(self.frames), 224, 224)
        #     frames2 = torch.zeros(batch_size, 1, len(self.frames), 224, 224)
        if len(self.frames) == 1:
            frames1 = numpy.zeros((batch_size, 1, 224, 224)).astype("float32")
            frames2 = numpy.zeros((batch_size, 1, 224, 224)).astype("float32")
        else:
            frames1 = numpy.zeros(
                (batch_size, len(self.frames), 224, 224, 3)).astype("float32")
            frames2 = numpy.zeros(
                (batch_size, len(self.frames), 224, 224, 3)).astype("float32")

        # labels = torch.zeros(batch_size, self.label_dims)
        labels = numpy.zeros((batch_size, self.label_dims)).astype("float32")
        for sample_i in range(batch_size):
            exp_i = idx[idx_range[sample_i]][0]
            frame_i = idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]
            video_name = self.data["exps"][exp_name]["video_name"][()].decode("utf-8")

            # get the desired frames.
            cap = cv2.VideoCapture(
                os.path.join(self.video_path, video_name))

            temp_feat1, temp_feat2 = self.get_frames(cap, frame_i)
            frames1[sample_i, :] = temp_feat1
            frames2[sample_i, :] = temp_feat2
            cap.release()
            label = self.data["exps"][exp_name]["labels"][frame_i, :]
            # labels[sample_i, :] = torch.from_numpy(label)
            labels[sample_i, :] = label

        return [frames1, frames2], labels

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        # print("a")
        idx_range = list(range(
            batch_id * self._pos_batch_size,
            (batch_id + 1) * self._pos_batch_size))

        # Loop over the pos_idx's. This array is shuffled at each epoch.
        # print(self.pos_idx[idx_range, :])
        # print("b")
        pos_features, pos_labels =\
            self.get_batch(self._pos_batch_size, self.pos_idx, idx_range)

        # figure out the negative samples.
        idx_range = list(range(
            batch_id * self._neg_batch_size,
            (batch_id + 1) * self._neg_batch_size))
        # print(idx_range)
        # neg_features = []
        neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)
        # to use the get_batch function, merge the frame and exp info, and
        # create a new idx_range. get_batch was mostly designed to grab frames
        # from the pos_idx, which is a list of all positives in the video.
        neg_exp_frame = numpy.asarray([neg_exps_idx, neg_frames_idx]).T
        # print(neg_exp_frame)
        neg_features, neg_labels = self.get_batch(
            self._neg_batch_size, neg_exp_frame, range(self._neg_batch_size))
        # print("c")
        # check to see if negative samples are too close to positive samples.
        if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
            import pdb; pdb.set_trace()

        # concatenate the positive and negative examples
        # Potential speed upgrade:
        # https://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
        # features = [
        #     torch.cat([pos, neg], dim=0)
        #     for pos, neg in zip(pos_features, neg_features)
        # ]
        features = [
            numpy.concatenate([pos, neg], axis=0)
            for pos, neg in zip(pos_features, neg_features)
        ]
        # for i in range(len(features)):
        #     # features[i] = Variable(features[i], requires_grad=True)
        #     features[i].requires_grad_(True)
        # labels = torch.cat([pos_labels, neg_labels])
        labels = numpy.concatenate([pos_labels, neg_labels], axis=0)
        # labels = Variable(labels, requires_grad=False)

        inputs = features + [labels]
        # if self.use_gpu >= 0:
        #     inputs = [
        #         input.cuda(self.use_gpu) for input in inputs
        #     ]

        return inputs

    def _find_neg_exp_frame(self, idx_range):
        """Helper function to find the negative samples."""
        # this is a weird way to do this... maybe doesn't make sense. This
        # enumerates all negative examples and searches each bout for the
        # deisred sample. So if we are looking for negative sample 10, and
        # the negative bouts are of length 2, 4, 8. Then we'd take the 4th
        # sample in the 3rd negative bout (of length 8), ie the 10th negative
        # sample of all the samples.
        # to improve the speed of this, sort the batch indexes. Then we can
        # search the bouts in one pass for each sample in the batch
        samples_idx = self.neg_idx[idx_range]
        samples_idx.sort()

        # row_i = 0
        # seen_frames = 0
        # remainder = idx
        # prev_bout = 0
        remainder = 0

        exps_idx = []
        frames_idx = []

        bout_cumsum = numpy.cumsum(self.neg_bouts[:, 3])
        for sample_i in samples_idx:
            # search for the bouts for the frame/exp info. Use the
            # numpy.cumsum function to figure out which bout contains the
            # nth negative sample
            try:
                idx = numpy.argwhere(bout_cumsum > sample_i)[0][0]
            except:
                print("Bad negative sample, greater than max number of negative frames")
                exit()
            exps_idx.append(self.neg_bouts[idx, 0])
            if idx == 0:
                num_seen_frames = 0
            else:
                num_seen_frames = bout_cumsum[idx - 1]
            remainder = sample_i - num_seen_frames
            frames_idx.append(self.neg_bouts[idx, 1] + remainder)

        # re-shuffle the samples? probably not needed in most cases... but
        # this would be a good place to do it.
        return exps_idx, frames_idx

    def check_negatives(self, neg_exps_idx, neg_frames_idx):
        """Make sure no negative frame is too close to a positive."""
        for neg_i in range(self._neg_batch_size):
            neg_exp = neg_exps_idx[neg_i]
            neg_frame = neg_frames_idx[neg_i]
            for i in range(self.num_pos):
                if self.pos_idx[i, 0] == neg_exp:
                    pos_frame = self.pos_idx[i, 1]
                    if numpy.abs(pos_frame - neg_frame) < 5:
                        return True
        return False

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # try:
        img = cv2.resize(img, (self.width, self.height))
        # except:
        # import pdb; pdb.set_trace()
        # pil_img = PIL.Image.fromarray(img)
        tensor_img = self.preproc(img)

        return tensor_img

