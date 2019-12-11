"""Helper functions to get hantman batches."""
from __future__ import print_function, division

import numpy
# import theano

# import threading
# import time
from . import DataLoader
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import PIL
import os
import cv2
import time

# python 2 vs 3 stuff...
import sys
if sys.version_info[0] < 3:
    import Queue as queue
else:
    import queue


# class HantmanSeqSampler:
#     def __init__(self, rng, data, mini_batch, seq_len=None, use_pool=False):
#         self.rng = rng
#         self.data = data
#         self.mini_batch = mini_batch
#         self.seq_len = seq_len
#         self.use_pool = use_pool
#         self.exp_list = data["exp_names"]


#     def reset(self):
#         self._batch_idx = 0
#         self._permuted_exps = self.rng.permutation(self.exp_list)
#         # a minibatch of training will go over the sequence length
#         # but a minibatch is a set of sequences to train. may not fit into
#         # the network by istelf.

#     # need a minibatch iterator, as well as a sequence iterator.


class HantmanFrameImageSampler():
    """HantmanFrameImageSampler

    Samples random frames stored on disk in folders.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, frame_path, mini_batch, max_workers=2,
                 max_queue=5, use_pool=True, gpu_id=-1):
        self.rng = rng
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id

        self.mini_batch = mini_batch
        # Because positives are so rare, we'll want to actively find positive
        # samples for the minibatch. This variable will deterimine how many
        # positive samples are needed per patch.
        self._pos_batch_size = int(mini_batch / 2)
        self._neg_batch_size = mini_batch - self._pos_batch_size

        self.exp_names = self.data["exp_names"]
        self.num_exp = len(self.exp_names)
        self.feat_dim = 100
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

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
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
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
        self.rng.shuffle(self.pos_idx)
        for i in range(self.num_batch):
            self.batch_idx.put(i)

        # create a random sequence of negative
        self.neg_idx = self.rng.choice(
            self.num_neg,
            (self._neg_batch_size * self.num_batch),
            replace=False)
        # import pdb; pdb.set_trace()
        # self.half_batch_idx = 0

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.batch_idx.empty():
            # the batch_idx is empty. an "epoch" has passed. resample the
            # positive index order.
            # print "\treset"
            self.reset()

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
            # print exp_name
            exp = self.data["exps"][exp_name]
            temp = numpy.argwhere(exp["org_labels"].value > 0)
            seq_len = exp["org_labels"].shape[0]

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
                bout_len = curr_pos - 5 - prev_idx
                # check for negative bout lengths. This can happen when the
                # behaviors are too close together.
                if bout_len > 0:
                    bout = numpy.array(
                        [i, prev_idx, curr_pos - 5, bout_len], ndmin=2)
                    neg_bouts.append(bout)
                prev_idx = min(curr_pos + 5, seq_len)
                if bout_len > 1000 and j != 0:
                    # this seems suspicious...
                    import pdb; pdb.set_trace()

                # check to make sure that the bout doesn't overlap with
                for k in range(temp_idx_mat.shape[0]):
                    if bout[0, 1] < temp_idx_mat[k][1] and bout[0, 2] > temp_idx_mat[k][1]:
                        import pdb; pdb.set_trace()

            # add the last negative bout (goes to the end of the video)
            if prev_idx != seq_len:
                neg_bouts.append(numpy.array(
                    [i, prev_idx, seq_len, seq_len - prev_idx], ndmin=2))
            else:
                import pdb; pdb.set_trace()
            pos_idx.append(temp_idx_mat)

        # concatenate all the pos_idx's into won giant array. First column
        # is the experiment index, second the frame number for that
        # experiment and then the third column is the behavior index.
        self.pos_idx = numpy.concatenate(pos_idx, axis=0)
        self.num_pos = self.pos_idx.shape[0]

        self.neg_bouts = numpy.concatenate(neg_bouts, axis=0)
        self.num_neg = self.neg_bouts[:, 3].sum()

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id * self._pos_batch_size,
            (batch_id + 1) * self._pos_batch_size))

        # for each key, create the positive sample array.
        self.feat_keys = ["moo"]
        # pos_features1 = []
        # pos_features2 = []
        # for key_i in range(len(self.feat_keys)):
        #     key = self.feat_keys[key_i]
        # temp_feat = numpy.zeros(
        #     (self._pos_batch_size, self.feat_dim[key_i]),
        #     dtype="float32"
        # )
        temp_feat1 = torch.zeros(self._pos_batch_size, 3, 224, 224)
        temp_feat2 = torch.zeros(self._pos_batch_size, 3, 224, 224)

        # Loop over the pos_idx's. This array is shuffled at each epoch.
        for sample_i in range(self._pos_batch_size):
            # open up the ith experiment
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            img_path = os.path.join(
                self.frame_path, exp_name, "frames", "%05d.jpg" % frame_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
            # temp_feat[sample_i, :] =\
            #     self.data["exps"][self.exp_names[exp_i]][key].value[frame_i, 0, :]
        pos_features = [temp_feat1, temp_feat2]
        # get the labels for these features.
        pos_labels = torch.zeros(self._pos_batch_size, 6)
        # positive sample labels
        for sample_i in range(self._pos_batch_size):
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
            pos_labels[sample_i, :] = torch.from_numpy(label)
            # # print(label)
            # label = numpy.argwhere(label)
            # # if label.size != 1:
            # #     import pdb; pdb.set_trace()
            # pos_labels[sample_i] = label[0][0]

        # figure out the negative samples.
        idx_range = list(range(
            batch_id * self._neg_batch_size,
            (batch_id + 1) * self._neg_batch_size))
        # neg_features = []
        neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)

        temp_feat1 = torch.zeros(self._neg_batch_size, 3, 224, 224)
        temp_feat2 = torch.zeros(self._neg_batch_size, 3, 224, 224)
        # loop over the negative examples
        for sample_i in range(self._neg_batch_size):
            # temp_feat[sample_i, 0] = exps_idx[sample_i]
            # temp_feat[sample_i, 1] = frames_idx[sample_i]
            neg_exps_idx_i = neg_exps_idx[sample_i]
            neg_frames_idx_i = neg_frames_idx[sample_i]

            exp_name = self.exp_names[neg_exps_idx_i]

            img_path = os.path.join(
                self.frame_path, exp_name, "frames",
                "%05d.jpg" % neg_frames_idx_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
        neg_features = [temp_feat1, temp_feat2]

        # not using convovled labels... soo all zero?
        neg_labels = torch.zeros(self._neg_batch_size, 6)
        # for sample_i in range(self._neg_batch_size):
        #     # exp_i = self.neg_idx[idx_range[sample_i]][0]
        #     # frame_i = self.neg_idx[idx_range[sample_i]][1]
        #     # label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
        #     # neg_labels[sample_i, :] = torch.from_numpy(label)
        #     # just set it to class 6... or doing MSE?
        #     # neg_labels[sample_i] = 6
        #     # self.data["exps"][self.exp_names[exp_i]]["labels"].value[frame_i, 0, :]

        # check to see if negative samples are too close to positive samples.
        if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
            import pdb; pdb.set_trace()

        # concatenate the positive and negative examples
        # Potential speed upgrade:
        # https://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
        # features = [
        #     numpy.concatenate([pos, neg], axis=0)
        #     for pos, neg in zip(pos_features, neg_features)
        # ]
        # labels = numpy.concatenate([pos_labels, neg_labels], axis=0)
        features = [
            torch.cat([pos, neg], dim=0)
            for pos, neg in zip(pos_features, neg_features)
        ]
        labels = torch.cat([pos_labels, neg_labels])
        # inputs = features + [labels]
        # inputs = [inputs[0][-1:, :], inputs[1][-1:, :], inputs[2][-1:]]
        # labels = self._convert_labels(labels)

        feat_var = [
            Variable(var, requires_grad=True) for var in features
        ]
        label_var = [Variable(labels, requires_grad=False)]
        inputs = feat_var + label_var
        if self.use_gpu >= 0:
            inputs = [
                input.cuda(self.use_gpu) for input in inputs
            ]

        return inputs

    def _find_neg_exp_frame(self, idx_range):
        """Helper function to find the negative samples."""
        # to improve the speed of this, sort the batch indexes. Then we can
        # search the bouts in one pass for each sample in the batch
        samples_idx = self.neg_idx[idx_range]
        samples_idx.sort()

        row_i = 0
        seen_frames = 0
        # remainder = idx
        prev_bout = 0
        remainder = 0

        exps_idx = []
        frames_idx = []

        for sample_i in samples_idx:
            # if sample_i == 9404:
            #     import pdb; pdb.set_trace()
            remainder = sample_i - seen_frames
            prev_bout = 0
            while seen_frames < self.num_neg:
                bout_frames = self.neg_bouts[row_i, 3]
                remainder = remainder - prev_bout
                if remainder < bout_frames:
                    # found the bout?
                    # if sample_i == 87022:
                    #     import pdb; pdb.set_trace()
                    # if self.neg_bouts[row_i, 0] == 19 and\
                    #         (self.neg_bouts[row_i, 1] + remainder) == 334:
                    #     import pdb; pdb.set_trace()
                    exps_idx.append(self.neg_bouts[row_i, 0])
                    frames_idx.append(self.neg_bouts[row_i, 1] + remainder)
                    break
                seen_frames += bout_frames
                prev_bout = bout_frames
                row_i += 1
            # if we get here bad things?
            if seen_frames > self.num_neg:
                import pdb; pdb.set_trace()

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
        # split the image into two frames
        half_width = int(img.shape[1] / 2)
        frame1 = img[:, :half_width, :]
        frame2 = img[:, half_width:, :]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)

        return tensor_img1, tensor_img2


class HantmanVideoFrameSampler():
    """HantmanVideoFrameSampler

    Samples full videos in sequential order.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, hdf_data, frame_path, max_workers=2,
                 max_queue=5, use_pool=True, gpu_id=-1):
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id

        self.exp_names = self.data["exp_names"]
        self.num_exp = len(self.exp_names)
        self.feat_dim = 100
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

        # compute the number of "batches" that can be created with the number
        # of positive examples.
        # Ignore remainders? Assume leftovers will be seen by the network
        # eventually.
        # self.num_batch = int(self.num_pos / self._pos_batch_size)
        # set the queue size, for the number of videos to store?
        self.num_batch = self.num_exp
        self.vid_idx = queue.Queue(self.num_batch)
        # setup the queue.
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
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
        # self.rng.shuffle(self.pos_idx)
        # for i in range(self.num_batch):
        #     self.batch_idx.put(i)
        for i in range(self.num_batch):
            self.vid_idx.put(i)

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # split the image into two frames
        half_width = int(img.shape[1] / 2)
        frame1 = img[:, :half_width, :]
        frame2 = img[:, half_width:, :]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)

        return tensor_img1, tensor_img2

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.vid_idx.empty():
            # the vid_dix is empty.
            print("\treset")
            self.reset()

        return minibatch

    def batch_sampler(self):
        exp_i = self.vid_idx.get()
        exp_name = self.exp_names[exp_i]

        num_frames = self.data["exps"][exp_name]["org_labels"].shape[0]
        num_frames = min(num_frames, 1500)

        temp_feat1 = torch.zeros(num_frames, 3, 224, 224)
        temp_feat2 = torch.zeros(num_frames, 3, 224, 224)
        # labels = torch.zeros(num_frames, 6)
        for frame_i in range(num_frames):
            img_path = os.path.join(
                self.frame_path, exp_name, "frames", "%05d.jpg" % frame_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)
            temp_feat1[frame_i, :, :, :] = tensor1
            temp_feat2[frame_i, :, :, :] = tensor2

        label = self.data["exps"][exp_name]["org_labels"].value
        labels = torch.from_numpy(label)
        # label = numpy.argwhere(label)
        # if label.size == 0:
        #      labels[frame_i] = 6
        # else:
        #     labels[frame_i] = label[0][0]

        # feat_var = [
        #     Variable(var, requires_grad=True) for var in [temp_feat1, temp_feat2]
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        # inputs = feat_var + label_var
        # if self.use_gpu is True:
        #     inputs = [
        #         input.cuda() for input in inputs
        #     ]
        inputs = [temp_feat1, temp_feat2, labels, exp_name]

        return inputs


class HantmanVideoSampler():
    """Video Sampler."""
    def __init__(self, rng, hdf_data, batch_size, seq_len, feat_keys,
                 use_gpu=-1, use_threads=False):
        if rng is None:
            self.rng = numpy.random.RandomState()
        else:
            self.rng = rng

        self.data = hdf_data
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.feat_keys = feat_keys
        self.seq_len = seq_len
        self.exp_names = self.data["exp_names"].value
        self.use_threads = use_threads

        # setup some space for the features, so that they can be dumped onto
        # the GPU.
        self._init_space()

        # batch book keeping information.
        self.samples_idx = 0
        self.samples = []
        self.reset()

    def reset(self, use_rand=True):
        """Reset the sampler (after an epoch)"""
        if use_rand is True:
            self.samples = self.rng.permutation(self.exp_names)
        else:
            self.samples = self.exp_names
        self.samples_idx = 0

    def _init_space(self):
        # initialize the space for featuers.
        self.feats = []
        for key in self.feat_keys:
            temp_feat = self.data["exps"][self.exp_names[0]][key]
            (t1, t2, feat_size) = temp_feat.shape

            # the default RNN behavior is to have the batch dimension
            # be the second one.
            self.feats.append(
                numpy.zeros(
                    (self.seq_len, self.batch_size, feat_size),
                    dtype=numpy.float32
                )
            )
        # create the label and mask spaces
        temp_feat = self.data["exps"][self.exp_names[0]]["labels"]
        (t1, t2, num_labels) = temp_feat.shape
        self.labels = numpy.zeros(
            (self.seq_len, self.batch_size, num_labels),
            dtype=numpy.float32
        )
        self.org_labels = numpy.zeros(
            (self.seq_len, self.batch_size, num_labels),
            dtype=numpy.float32
        )
        self.mask = numpy.zeros(
            (self.seq_len, self.batch_size, num_labels),
            dtype=numpy.float32
        )

    def get_mini_batch(self):
        """Get a mini batch."""
        exp_names = []
        tic = time.time()
        for i in range(self.batch_size):
            # loop over the feature keys
            self._fill_features(self.samples_idx, i)
            exp_names.append(self.samples[self.samples_idx])
            self.samples_idx += 1
        ram_time = time.time() - tic

        # package the data for processing.
        data_blob = self._package_blob(exp_names, ram_time)
        return data_blob

    def _package_blob(self, exp_names, ram_time):
        """Package the output blob."""
        # put the data onto the gpu.
        # Assumes that self.feats, self.labels, self.mask are properly
        # populated.
        tic = time.time()
        feats, labels, mask = self._create_tensors()
        gpu_time = time.time() - tic

        data_blob = {
            "exp_names": exp_names,
            "features": feats,
            "labels": labels,
            "org_labels": self.org_labels,
            "mask": mask,
            "ram_time": ram_time,
            "gpu_time": gpu_time
        }
        return data_blob

    def _create_tensors(self):
        """Create pytorch tensors."""
        feats = [
            Variable(torch.Tensor(feat))
            for feat in self.feats
        ]
        labels = Variable(torch.Tensor(self.labels))
        # org_labels are not used with torch tensors.
        mask = Variable(torch.Tensor(self.mask))

        if self.use_gpu > -1:
            # put onto the gpu.
            feats = [
                feat.cuda()
                for feat in feats
            ]
            labels = labels.cuda()
            # org_labels are not used with torch tensors.
            mask = mask.cuda()

        return feats, labels, mask

    def _fill_features(self, queue_idx, batch_idx):
        """Fill up the features."""
        idx = 0
        for key in self.feat_keys:
            temp_feat = self.data["exps"][self.samples[queue_idx]][key]
            seq_len = temp_feat.shape[0]
            self.feats[idx][:seq_len, batch_idx, :] = temp_feat[:, 0, :]
            idx += 1

        # fill the mask
        self.mask[:seq_len, batch_idx] = 1
        # fill the labels
        temp_labels = self.data["exps"][self.samples[queue_idx]]["labels"][:, 0, :]
        self.labels[:seq_len, batch_idx, :] = temp_labels
        temp_labels = self.data["exps"][self.samples[queue_idx]]["org_labels"]
        self.org_labels[:seq_len, batch_idx, :] = temp_labels

    def get_rest(self):
        # when batch sizes don't fit, helper function to just get the remaining
        # samples.
        tic = time.time()
        num_left = len(self.samples) - self.samples_idx

        feats = [
            numpy.zeros(
                (self.seq_len, num_left, self.feats[i].shape[2]),
                dtype=numpy.float32)
            for i in range(len(self.feat_keys))
        ]

        exp_names = []
        labels = numpy.zeros(
            (self.seq_len, self.batch_size, self.labels.shape[2]),
            dtype=numpy.float32)
        org_labels = numpy.zeros(
            (self.seq_len, self.batch_size, self.labels.shape[2]),
            dtype=numpy.float32)
        mask = numpy.zeros(
            (self.seq_len, self.batch_size, self.labels.shape[2]),
            dtype=numpy.float32)

        for i in range(num_left):
            idx = self.samples[self.samples_idx + i]
            key_idx = 0
            for key in self.feat_keys:
                temp_feat = self.data["exps"][idx][key]
                seq_len = temp_feat.shape[0]
                feats[key_idx][:seq_len, i, :] = temp_feat[:, 0, :]
                key_idx += 1

            labels[:seq_len, i, :] = self.data["exps"][idx]["labels"][:, 0, :]
            org_labels[:seq_len, i, :] = self.data["exps"][idx]["org_labels"]
            mask[:seq_len, i, :] = 1
            exp_names.append(
                idx
            )
        ram_time = time.time() - tic
        # package the data for processing.
        data_blob = self._package_blob(exp_names, ram_time)
        return data_blob

    def __iter__(self):
        # needed to make the class into an iteratable class.
        return self

    def next(self):
        if (self.samples_idx + self.batch_size) <= len(self.exp_names):
            return self.get_mini_batch()
        else:
            raise StopIteration()


class HantmanSeqFrameSampler():
    """HantmanFrameImageSampler

    Samples random frames stored on disk in folders.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, frame_path, mini_batch, max_workers=2,
                 max_queue=5, use_pool=True, gpu_id=-1):
        self.rng = rng
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id

        self.mini_batch = mini_batch
        # Because positives are so rare, we'll want to actively find positive
        # samples for the minibatch. This variable will deterimine how many
        # positive samples are needed per patch.
        self._pos_batch_size = int(mini_batch / 2)
        self._neg_batch_size = mini_batch - self._pos_batch_size

        self.exp_names = self.data["exp_names"]
        self.num_exp = len(self.exp_names)
        self.feat_dim = 100
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

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
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # initialize the workers if necessary.
        if self.use_pool is True:
            # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        # self.half_batch_idx = 0

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.batch_idx.empty():
            # the batch_idx is empty. an "epoch" has passed. resample the
            # positive index order.
            # print "\treset"
            self.reset()

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
            # print exp_name
            exp = self.data["exps"][exp_name]
            temp = numpy.argwhere(exp["org_labels"].value > 0)
            seq_len = exp["org_labels"].shape[0]

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
                #     import pdb; pdb.set_trace()

                # check to make sure that the bout doesn't overlap with
                for k in range(temp_idx_mat.shape[0]):
                    if bout[0, 1] < temp_idx_mat[k][1] and bout[0, 2] > temp_idx_mat[k][1]:
                        import pdb; pdb.set_trace()

            # add the last negative bout (goes to the end of the video)
            if prev_idx != seq_len:
                neg_bouts.append(numpy.array(
                    [i, prev_idx, seq_len, seq_len - prev_idx], ndmin=2))
            else:
                import pdb; pdb.set_trace()
            pos_idx.append(temp_idx_mat)

        # concatenate all the pos_idx's into won giant array. First column
        # is the experiment index, second the frame number for that
        # experiment and then the third column is the behavior index.
        self.pos_idx = numpy.concatenate(pos_idx, axis=0)
        self.num_pos = self.pos_idx.shape[0]

        self.neg_bouts = numpy.concatenate(neg_bouts, axis=0)
        self.num_neg = self.neg_bouts[:, 3].sum()

    def _load_seq(self, exp_name, frame_i):
        """Load the sub sequence."""

        feat1 = torch.zeros(1, 10, 224, 224)
        feat2 = torch.zeros(1, 10, 224, 224)
        frames = range(frame_i - 5, frame_i + 5)
        if numpy.any(numpy.array(frames) < 0):
            # import pdb; pdb.set_trace()
            for i in range(len(frames)):
                if frames[i] < 0:
                    frames[i] = 0
        idxs = range(0, 10)
        for i, frame in zip(idxs, frames):
            img_path = os.path.join(
                self.frame_path, exp_name, "frames", "%05d.jpg" % frame
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)

            feat1[:, i, :, :] = tensor1
            feat2[:, i, :, :] = tensor2
        return feat1, feat2

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id * self._pos_batch_size,
            (batch_id + 1) * self._pos_batch_size))

        # for each key, create the positive sample array.
        self.feat_keys = ["moo"]
        # pos_features1 = []
        # pos_features2 = []
        # for key_i in range(len(self.feat_keys)):
        #     key = self.feat_keys[key_i]
        # temp_feat = numpy.zeros(
        #     (self._pos_batch_size, self.feat_dim[key_i]),
        #     dtype="float32"
        # )
        temp_feat1 = torch.zeros(self._pos_batch_size, 1, 10, 224, 224)
        temp_feat2 = torch.zeros(self._pos_batch_size, 1, 10, 224, 224)

        # Loop over the pos_idx's. This array is shuffled at each epoch.
        for sample_i in range(self._pos_batch_size):
            # open up the ith experiment
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            tensor1, tensor2 = self._load_seq(exp_name, frame_i)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
            # temp_feat[sample_i, :] =\
            #     self.data["exps"][self.exp_names[exp_i]][key].value[frame_i, 0, :]
        pos_features = [temp_feat1, temp_feat2]
        # get the labels for these features.
        pos_labels = torch.zeros(self._pos_batch_size, 6)
        # positive sample labels
        for sample_i in range(self._pos_batch_size):
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
            pos_labels[sample_i, :] = torch.from_numpy(label)
            # # print(label)
            # label = numpy.argwhere(label)
            # # if label.size != 1:
            # #     import pdb; pdb.set_trace()
            # pos_labels[sample_i] = label[0][0]

        # figure out the negative samples.
        idx_range = list(range(
            batch_id * self._neg_batch_size,
            (batch_id + 1) * self._neg_batch_size))
        # neg_features = []
        neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)

        temp_feat1 = torch.zeros(self._neg_batch_size, 1, 10, 224, 224)
        temp_feat2 = torch.zeros(self._neg_batch_size, 1, 10, 224, 224)
        # loop over the negative examples
        for sample_i in range(self._neg_batch_size):
            # temp_feat[sample_i, 0] = exps_idx[sample_i]
            # temp_feat[sample_i, 1] = frames_idx[sample_i]
            neg_exps_idx_i = neg_exps_idx[sample_i]
            neg_frames_idx_i = neg_frames_idx[sample_i]

            exp_name = self.exp_names[neg_exps_idx_i]

            tensor1, tensor2 = self._load_seq(exp_name, neg_frames_idx_i)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
        neg_features = [temp_feat1, temp_feat2]

        # not using convovled labels... soo all zero?
        neg_labels = torch.zeros(self._neg_batch_size, 6)

        # check to see if negative samples are too close to positive samples.
        if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
            import pdb; pdb.set_trace()

        # concatenate the positive and negative examples
        # Potential speed upgrade:
        # https://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
        # features = [
        #     numpy.concatenate([pos, neg], axis=0)
        #     for pos, neg in zip(pos_features, neg_features)
        # ]
        # labels = numpy.concatenate([pos_labels, neg_labels], axis=0)
        features = [
            torch.cat([pos, neg], dim=0)
            for pos, neg in zip(pos_features, neg_features)
        ]
        labels = torch.cat([pos_labels, neg_labels])
        labels = self._convert_labels(labels)
        # inputs = features + [labels]
        # inputs = [inputs[0][-1:, :], inputs[1][-1:, :], inputs[2][-1:]]

        feat_var = [
            Variable(var, requires_grad=True) for var in features
        ]
        label_var = [Variable(labels, requires_grad=False)]
        inputs = feat_var + label_var
        if self.use_gpu >= 0:
            inputs = [
                input.cuda(self.use_gpu) for input in inputs
            ]

        return inputs

    def _convert_labels(self, labels):
        idx = labels.nonzero()
        # assume half batch is positive
        num_pos = int(labels.size(0) / 2)
        curr_idx = 0
        temp = []
        for i in range(idx.size(0)):
            if idx[i, 0] == curr_idx:
                temp.append(idx[i, 1])
                curr_idx += 1
        pos_labels = torch.LongTensor(temp)
        relabel = torch.LongTensor(
            [6 for i in range(num_pos, labels.size(0))]
        )
        relabel = torch.cat([pos_labels, relabel], dim=0)
        # if idx.size(0) != 2:
        #     import pdb; pdb.set_trace()
        # if relabel.size(0) != labels.size(0):
        #     import pdb; pdb.set_trace()
        return relabel

    def _find_neg_exp_frame(self, idx_range):
        """Helper function to find the negative samples."""
        # to improve the speed of this, sort the batch indexes. Then we can
        # search the bouts in one pass for each sample in the batch
        samples_idx = self.neg_idx[idx_range]
        samples_idx.sort()

        row_i = 0
        seen_frames = 0
        # remainder = idx
        prev_bout = 0
        remainder = 0

        exps_idx = []
        frames_idx = []

        for sample_i in samples_idx:
            # if sample_i == 9404:
            #     import pdb; pdb.set_trace()
            remainder = sample_i - seen_frames
            prev_bout = 0
            while seen_frames < self.num_neg:
                bout_frames = self.neg_bouts[row_i, 3]
                remainder = remainder - prev_bout
                if remainder < bout_frames:
                    # found the bout?
                    # if sample_i == 87022:
                    #     import pdb; pdb.set_trace()
                    # if self.neg_bouts[row_i, 0] == 19 and\
                    #         (self.neg_bouts[row_i, 1] + remainder) == 334:
                    #     import pdb; pdb.set_trace()
                    exps_idx.append(self.neg_bouts[row_i, 0])
                    frames_idx.append(self.neg_bouts[row_i, 1] + remainder)
                    break
                seen_frames += bout_frames
                prev_bout = bout_frames
                row_i += 1
            # if we get here bad things?
            if seen_frames > self.num_neg:
                import pdb; pdb.set_trace()

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
        # split the image into two frames
        half_width = int(img.shape[1] / 2)
        frame1 = img[:, :half_width, 0]
        frame2 = img[:, half_width:, 0]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)
        # import pdb; pdb.set_trace()
        return tensor_img1, tensor_img2


class HantmanFrameGrabber():
    """Sample individual frames from disk."""
    def __init__(self, frame_dir, batch_size=50, dim=(260, 704, 3),
                 use_threads=False):
        """Initialize the frame grabber."""
        self.frame_dir = frame_dir

        # grabs frames in chunks, set by batch_size.
        self.batch_size = batch_size
        # indexing info.
        self.exp_names = os.listdir(self.frame_dir)
        self.exp_names.sort()

        self.exp_name = ""
        self.exp_dir = ""
        self.frame_idx = -1
        self.num_frames = -1
        self.dims = (self.batch_size, dim[0], dim[1], dim[2])

        self.cache = numpy.zeros(self.dims, dtype="uint8")

    def set_exp(self, exp_name):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(
            self.frame_dir, self.exp_name, "frames"
        )
        self.num_frames = len([
            fname for fname in os.listdir(self.exp_dir)
            if fname.endswith(".jpg")
        ])

        self.frame_idx = 0

    def get_batch(self):
        """Grab a batch of data."""
        if self.exp_name is "":
            return (-1, self.cache)

        # loop over and get the frames
        # frame_i = self.frame_idx
        rem_frames = self.num_frames - self.frame_idx

        batch_size = min(self.batch_size, rem_frames)
        for i in range(batch_size):
            # print(self.frame_idx)
            img_path = os.path.join(
                self.exp_dir, "%05d.jpg" % self.frame_idx
            )
            img = cv2.imread(img_path)
            # cv2.imshow("moo", img)
            # cv2.waitKey(10)
            try:
                self.cache[i, :, :, :] = img
            except:
                import pdb; pdb.set_trace()

            self.frame_idx = self.frame_idx + 1
        # import pdb; pdb.set_trace()
        return (batch_size, self.cache)

    def __iter__(self):
        """Needed for an iterable object."""
        return self

    def next(self):
        """Iterator function."""
        if self.exp_name is "":
            raise StopIteration

        if self.frame_idx >= self.num_frames:
            raise StopIteration

        (size, batch) = self.get_batch()

        return batch[:size, :, :, :]


class HantmanVideoSeqFrameSampler():
    """HantmanVideoSeqFrameSampler

    Samples full videos in sequential order.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, hdf_data, frame_path, max_workers=2,
                 max_queue=5, use_pool=True, gpu_id=-1):
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id

        self.exp_names = self.data["exp_names"]
        self.num_exp = len(self.exp_names)
        self.feat_dim = 100
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

        # compute the number of "batches" that can be created with the number
        # of positive examples.
        # Ignore remainders? Assume leftovers will be seen by the network
        # eventually.
        # self.num_batch = int(self.num_pos / self._pos_batch_size)
        # set the queue size, for the number of videos to store?
        self.num_batch = self.num_exp
        self.vid_idx = queue.Queue(self.num_batch)
        # setup the queue.
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
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
        # self.rng.shuffle(self.pos_idx)
        # for i in range(self.num_batch):
        #     self.batch_idx.put(i)
        for i in range(self.num_batch):
            self.vid_idx.put(i)

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # split the image into two frames
        half_width = int(img.shape[1] / 2)
        frame1 = img[:, :half_width, 0]
        frame2 = img[:, half_width:, 0]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)
        # import pdb; pdb.set_trace()
        return tensor_img1, tensor_img2

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.vid_idx.empty():
            # the vid_dix is empty.
            print("\treset")
            self.reset()

        return minibatch

    def batch_sampler(self):
        exp_i = self.vid_idx.get()
        exp_name = self.exp_names[exp_i]

        num_frames = self.data["exps"][exp_name]["org_labels"].shape[0]
        num_frames = min(num_frames, 1500)

        temp_feat1 = torch.zeros(num_frames, 1, 224, 224)
        temp_feat2 = torch.zeros(num_frames, 1, 224, 224)
        # labels = torch.zeros(num_frames, 6)
        for frame_i in range(num_frames):
            img_path = os.path.join(
                self.frame_path, exp_name, "frames", "%05d.jpg" % frame_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)
            temp_feat1[frame_i, :, :, :] = tensor1
            temp_feat2[frame_i, :, :, :] = tensor2

        label = self.data["exps"][exp_name]["org_labels"].value
        labels = torch.from_numpy(label)
        # label = numpy.argwhere(label)
        # if label.size == 0:
        #      labels[frame_i] = 6
        # else:
        #     labels[frame_i] = label[0][0]

        # feat_var = [
        #     Variable(var, requires_grad=True) for var in [temp_feat1, temp_feat2]
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        # inputs = feat_var + label_var
        # if self.use_gpu is True:
        #     inputs = [
        #         input.cuda() for input in inputs
        #     ]
        inputs = [temp_feat1, temp_feat2, labels, exp_name]

        return inputs


class HantmanBWFrameImageSampler():
    """HantmanFrameImageSampler

    Samples random frames stored on disk in folders.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, frame_path, mini_batch, max_workers=2,
                 max_queue=5, use_pool=True, gpu_id=-1, mean=0.5, std=0.2):
        self.rng = rng
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id

        self.mini_batch = mini_batch
        # Because positives are so rare, we'll want to actively find positive
        # samples for the minibatch. This variable will deterimine how many
        # positive samples are needed per patch.
        self._pos_batch_size = int(mini_batch / 2)
        self._neg_batch_size = mini_batch - self._pos_batch_size

        self.exp_names = self.data["exp_names"]
        self.num_exp = len(self.exp_names)
        self.feat_dim = 100
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

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
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=mean,
                                         std=std)

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
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
        self.rng.shuffle(self.pos_idx)
        for i in range(self.num_batch):
            self.batch_idx.put(i)

        # create a random sequence of negative
        self.neg_idx = self.rng.choice(
            self.num_neg,
            (self._neg_batch_size * self.num_batch),
            replace=False)
        # import pdb; pdb.set_trace()
        # self.half_batch_idx = 0

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.batch_idx.empty():
            # the batch_idx is empty. an "epoch" has passed. resample the
            # positive index order.
            # print "\treset"
            self.reset()

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
            # print exp_name
            exp = self.data["exps"][exp_name]
            temp = numpy.argwhere(exp["org_labels"].value > 0)
            seq_len = exp["org_labels"].shape[0]

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
                bout_len = curr_pos - 5 - prev_idx
                # check for negative bout lengths. This can happen when the
                # behaviors are too close together.
                if bout_len > 0:
                    bout = numpy.array(
                        [i, prev_idx, curr_pos - 5, bout_len], ndmin=2)
                    neg_bouts.append(bout)
                prev_idx = min(curr_pos + 5, seq_len)
                if bout_len > 1000 and j != 0:
                    # this seems suspicious...
                    import pdb; pdb.set_trace()

                # check to make sure that the bout doesn't overlap with
                for k in range(temp_idx_mat.shape[0]):
                    if bout[0, 1] < temp_idx_mat[k][1] and bout[0, 2] > temp_idx_mat[k][1]:
                        import pdb; pdb.set_trace()

            # add the last negative bout (goes to the end of the video)
            if prev_idx != seq_len:
                neg_bouts.append(numpy.array(
                    [i, prev_idx, seq_len, seq_len - prev_idx], ndmin=2))
            else:
                import pdb; pdb.set_trace()
            pos_idx.append(temp_idx_mat)

        # concatenate all the pos_idx's into won giant array. First column
        # is the experiment index, second the frame number for that
        # experiment and then the third column is the behavior index.
        self.pos_idx = numpy.concatenate(pos_idx, axis=0)
        self.num_pos = self.pos_idx.shape[0]

        self.neg_bouts = numpy.concatenate(neg_bouts, axis=0)
        self.num_neg = self.neg_bouts[:, 3].sum()

    def batch_sampler(self):
        batch_id = self.batch_idx.get()
        # print "worker %d: loaded data, waiting to put..." % self.threadid
        # now actually get the batch
        idx_range = list(range(
            batch_id * self._pos_batch_size,
            (batch_id + 1) * self._pos_batch_size))

        # for each key, create the positive sample array.
        self.feat_keys = ["moo"]
        # pos_features1 = []
        # pos_features2 = []
        # for key_i in range(len(self.feat_keys)):
        #     key = self.feat_keys[key_i]
        # temp_feat = numpy.zeros(
        #     (self._pos_batch_size, self.feat_dim[key_i]),
        #     dtype="float32"
        # )
        temp_feat1 = torch.zeros(self._pos_batch_size, 1, 224, 224)
        temp_feat2 = torch.zeros(self._pos_batch_size, 1, 224, 224)

        # Loop over the pos_idx's. This array is shuffled at each epoch.
        for sample_i in range(self._pos_batch_size):
            # open up the ith experiment
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            img_path = os.path.join(
                self.frame_path, exp_name, "frames", "%05d.jpg" % frame_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
            # temp_feat[sample_i, :] =\
            #     self.data["exps"][self.exp_names[exp_i]][key].value[frame_i, 0, :]
        pos_features = [temp_feat1, temp_feat2]
        # get the labels for these features.
        pos_labels = torch.zeros(self._pos_batch_size, 6)
        # positive sample labels
        for sample_i in range(self._pos_batch_size):
            exp_i = self.pos_idx[idx_range[sample_i]][0]
            frame_i = self.pos_idx[idx_range[sample_i]][1]
            exp_name = self.exp_names[exp_i]

            label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
            pos_labels[sample_i, :] = torch.from_numpy(label)
            # # print(label)
            # label = numpy.argwhere(label)
            # # if label.size != 1:
            # #     import pdb; pdb.set_trace()
            # pos_labels[sample_i] = label[0][0]

        # figure out the negative samples.
        idx_range = list(range(
            batch_id * self._neg_batch_size,
            (batch_id + 1) * self._neg_batch_size))
        # neg_features = []
        neg_exps_idx, neg_frames_idx = self._find_neg_exp_frame(idx_range)

        temp_feat1 = torch.zeros(self._neg_batch_size, 3, 224, 224)
        temp_feat2 = torch.zeros(self._neg_batch_size, 3, 224, 224)
        # loop over the negative examples
        for sample_i in range(self._neg_batch_size):
            # temp_feat[sample_i, 0] = exps_idx[sample_i]
            # temp_feat[sample_i, 1] = frames_idx[sample_i]
            neg_exps_idx_i = neg_exps_idx[sample_i]
            neg_frames_idx_i = neg_frames_idx[sample_i]

            exp_name = self.exp_names[neg_exps_idx_i]

            img_path = os.path.join(
                self.frame_path, exp_name, "frames",
                "%05d.jpg" % neg_frames_idx_i
            )
            img = cv2.imread(img_path)
            tensor1, tensor2 = self.preprocess_img(img)

            temp_feat1[sample_i, :] = tensor1
            temp_feat2[sample_i, :] = tensor2
        neg_features = [temp_feat1, temp_feat2]

        # not using convovled labels... soo all zero?
        neg_labels = torch.zeros(self._neg_batch_size, 6)
        # for sample_i in range(self._neg_batch_size):
        #     # exp_i = self.neg_idx[idx_range[sample_i]][0]
        #     # frame_i = self.neg_idx[idx_range[sample_i]][1]
        #     # label = self.data["exps"][exp_name]["org_labels"].value[frame_i, :]
        #     # neg_labels[sample_i, :] = torch.from_numpy(label)
        #     # just set it to class 6... or doing MSE?
        #     # neg_labels[sample_i] = 6
        #     # self.data["exps"][self.exp_names[exp_i]]["labels"].value[frame_i, 0, :]

        # check to see if negative samples are too close to positive samples.
        if self.check_negatives(neg_exps_idx, neg_frames_idx) is True:
            import pdb; pdb.set_trace()

        # concatenate the positive and negative examples
        # Potential speed upgrade:
        # https://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
        # features = [
        #     numpy.concatenate([pos, neg], axis=0)
        #     for pos, neg in zip(pos_features, neg_features)
        # ]
        # labels = numpy.concatenate([pos_labels, neg_labels], axis=0)
        features = [
            torch.cat([pos, neg], dim=0)
            for pos, neg in zip(pos_features, neg_features)
        ]
        labels = torch.cat([pos_labels, neg_labels])
        # inputs = features + [labels]
        # inputs = [inputs[0][-1:, :], inputs[1][-1:, :], inputs[2][-1:]]
        # labels = self._convert_labels(labels)

        feat_var = [
            Variable(var, requires_grad=True) for var in features
        ]
        label_var = [Variable(labels, requires_grad=False)]
        inputs = feat_var + label_var
        if self.use_gpu >= 0:
            inputs = [
                input.cuda(self.use_gpu) for input in inputs
            ]

        return inputs

    def _find_neg_exp_frame(self, idx_range):
        """Helper function to find the negative samples."""
        # to improve the speed of this, sort the batch indexes. Then we can
        # search the bouts in one pass for each sample in the batch
        samples_idx = self.neg_idx[idx_range]
        samples_idx.sort()

        row_i = 0
        seen_frames = 0
        # remainder = idx
        prev_bout = 0
        remainder = 0

        exps_idx = []
        frames_idx = []

        for sample_i in samples_idx:
            # if sample_i == 9404:
            #     import pdb; pdb.set_trace()
            remainder = sample_i - seen_frames
            prev_bout = 0
            while seen_frames < self.num_neg:
                bout_frames = self.neg_bouts[row_i, 3]
                remainder = remainder - prev_bout
                if remainder < bout_frames:
                    # found the bout?
                    # if sample_i == 87022:
                    #     import pdb; pdb.set_trace()
                    # if self.neg_bouts[row_i, 0] == 19 and\
                    #         (self.neg_bouts[row_i, 1] + remainder) == 334:
                    #     import pdb; pdb.set_trace()
                    exps_idx.append(self.neg_bouts[row_i, 0])
                    frames_idx.append(self.neg_bouts[row_i, 1] + remainder)
                    break
                seen_frames += bout_frames
                prev_bout = bout_frames
                row_i += 1
            # if we get here bad things?
            if seen_frames > self.num_neg:
                import pdb; pdb.set_trace()

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
        # split the image into two frames
        half_width = int(img.shape[1] / 2)
        frame1 = img[:, :half_width, 0]
        frame2 = img[:, half_width:, 0]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)

        return tensor_img1, tensor_img2


class HantmanFrameSeqSampler():
    """HantmanVideoFrameSampler

    Samples full videos in sequential order.

    If use_pool is true, the sampler will be multithreaded and maybe faster.
    Unfortunately there's no guarentee that the pooled data loader is faster.
    """
    def __init__(self, rng, hdf_data, frame_path, seq_len, mini_batch,
                 max_workers=1, max_queue=5, mean=0.5, std=0.2,
                 use_pool=False, gpu_id=-1):
        self.data = hdf_data
        self.frame_path = frame_path
        self.use_pool = use_pool
        self.use_gpu = gpu_id
        self.mini_batch = mini_batch
        self.rng = rng
        self.seq_len = seq_len

        self.exp_names = self.data["exp_names"].value
        self.num_exp = len(self.exp_names)
        # [self.data["exps"][self.exp_names[0]][key].value.shape[2]
        #  for key in feat_keys]
        self.num_behav =\
            self.data["exps"][self.exp_names[0]]["labels"].value.shape[2]

        # compute the number of "batches" that can be created with the number
        # of positive examples.
        # Ignore remainders? Assume leftovers will be seen by the network
        # eventually.
        self.num_batch = self.num_exp // self.mini_batch

        # setup the queue.
        self.vid_idx = queue.Queue(self.num_batch)
        self.reset()

        # setup the preprocessing
        normalize = transforms.Normalize(mean=[mean],
                                         std=[std])

        self.preproc = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
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
        self.rng.shuffle(self.exp_names)
        for i in range(self.num_batch):
            self.vid_idx.put(i)
        # for i in range(self.num_batch):
        #     self.vid_idx.put(i)

    def preprocess_img(self, img):
        """Apply pytorch preprocessing."""
        # split the image into two frames
        half_width = int(img.shape[1] / 2)

        frame1 = img[:, :half_width, 0]
        frame2 = img[:, half_width:, 0]

        # resize to fit into the network.
        frame1 = cv2.resize(frame1, (224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        pil1 = PIL.Image.fromarray(frame1)
        tensor_img1 = self.preproc(pil1)

        pil2 = PIL.Image.fromarray(frame2)
        tensor_img2 = self.preproc(pil2)

        return tensor_img1, tensor_img2

    def get_minibatch(self):
        """Get a minibatch of data."""
        if self.use_pool is True:
            minibatch = self._pool.get()
        else:
            minibatch = self.batch_sampler()

        if self.vid_idx.empty():
            # the vid_dix is empty.
            print("\treset")
            self.reset()

        return minibatch

    def batch_sampler(self):
        print("sampling")
        batch_i = self.vid_idx.get()

        # feat1 = []
        # feat2 = []
        feats = []
        labels = []
        org_labels = []
        exp_names = []
        for sample_i in range(self.mini_batch):
            exp_idx = batch_i * self.mini_batch + sample_i
            exp_name = self.exp_names[exp_idx]
            # print(exp_name)
            num_frames = self.data["exps"][exp_name]["labels"].shape[0]
            num_frames = min(num_frames, self.seq_len)

            temp_feat1 = torch.zeros(num_frames, 1, 224, 224)
            temp_feat2 = torch.zeros(num_frames, 1, 224, 224)
            # labels = torch.zeros(num_frames, 6)
            for frame_i in range(num_frames):
                img_path = os.path.join(
                    self.frame_path, exp_name, "frames", "%05d.jpg" % frame_i
                )
                img = cv2.imread(img_path)
                tensor1, tensor2 = self.preprocess_img(img)
                temp_feat1[frame_i, :, :, :] = tensor1
                temp_feat2[frame_i, :, :, :] = tensor2

            # feat1.append(temp_feat1.cuda(self.use_gpu))
            # feat2.append(temp_feat2.cuda(self.use_gpu))
            feat_var = [
                Variable(var, requires_grad=True) for var in
                [temp_feat1.cuda(self.use_gpu), temp_feat2.cuda(self.use_gpu)]
            ]
            feats.append(feat_var)

            label =\
                self.data["exps"][exp_name]["labels"].value[:num_frames, :, :]
            labels.append(torch.from_numpy(label).cuda(self.use_gpu))

            org_label =\
                self.data["exps"][exp_name]["org_labels"].value[:num_frames, :]
            org_labels.append(torch.from_numpy(org_label))

            exp_names.append(exp_name)

        # feat_var = [
        #     Variable(var, requires_grad=True) for var in [temp_feat1, temp_feat2]
        # ]
        # label_var = [Variable(labels, requires_grad=False)]
        # inputs = feat_var + label_var
        # if self.use_gpu is True:
        #     inputs = [
        #         input.cuda() for input in inputs
        #     ]
        # inputs = [temp_feat1, temp_feat2, labels, exp_name]
        data_blob = {
            "exp_names": exp_names,
            "features": feats,
            "labels": labels,
            "org_labels": org_labels,
            # "ram_time": ram_time,
            # "gpu_time": gpu_time
        }

        return data_blob
