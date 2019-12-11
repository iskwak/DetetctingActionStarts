# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from queue import Queue
import threading
import i3d.i3d as i3d
import cv2
import time
import os
import h5py
import helpers.tf_videosamplers as videosamplers
import helpers.paths as paths
import sys
import helpers.sequences_helper2 as sequences_helper

_IMAGE_SIZE = 224
# _BATCH_SIZE = 25
_BATCH_SIZE = 1

# _SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_VIDEO_FRAMES = 64
_SAMPLE_PATHS = {
    'rgb': '/nrs/branson/kwaki/data/models/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': '/nrs/branson/kwaki/data/models/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': '/nrs/branson/kwaki/data/models/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': '/nrs/branson/kwaki/data/models/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/nrs/branson/kwaki/data/models/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': '/nrs/branson/kwaki/data/models/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = '/nrs/branson/kwaki/data/lists/label_map.txt'
_LABEL_MAP_PATH_600 = '/nrs/branson/kwaki/data/lists/data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string('filelist', '', 'Text file with list of experiments.')
# jtf.flags.DEFINE_integer('gpus', 0, 'GPU to use.')
tf.flags.DEFINE_integer('window_start', -12, 'Offset from the desired frame.')
tf.flags.DEFINE_integer('window_size', 25, 'Window size.')
tf.flags.DEFINE_integer('batch_size', 25, 'Batch size.')
tf.flags.DEFINE_string('frames', '',
                       'List of frames to use. automatically generated.')
tf.flags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
tf.flags.DEFINE_integer(
  "save_iterations", 10,
  "Number of iterations to save the network.")
tf.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate.')
tf.flags.DEFINE_integer('frame', 0, 'side or front.')

# flags from video sampler
tf.flags.DEFINE_boolean("use_pool", False, "Use a pool (threads) for the sampler")
tf.flags.DEFINE_integer("max_workers", 2, "Max number of workers.")
tf.flags.DEFINE_integer("max_queue", 5, "Maximum queue length.")
tf.flags.DEFINE_string("out_dir", None, "Output directory path.")
tf.flags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
tf.flags.DEFINE_string("display_dir", None, "display dir.")
tf.flags.DEFINE_string("video_dir", None, "video dir.")
tf.flags.DEFINE_integer("hantman_mini_batch", 10, "video dir.")
tf.flags.DEFINE_boolean("reweight", True, "reweight labels.")


def crop_frame(img, crop_size=224):
  # resize version
  img = np.array(cv2.resize(np.array(img),(crop_size, crop_size))).astype(np.float32)
  # also rescale the pixel values
  # the i3d network expects values between -1 and 1
  # img = img - img.min()
  # img = img / img.max() * 2
  # img = img - 1
  img = img / 255 * 2 - 1
  img = img[:, :, :2]
  return img


def create_initial_batch(cap, crop_size, window_size, window_start):
  """Create the initial batch."""
  # loop for window start frames and add the first frame that many times.
  # then add window size - window start new frames.

  # print("\t\tFirst read")
  retval, frame = cap.read()
  # fill the list with num_frames_per_clip - 1.
  view1 = frame[:, :352, :]
  view2 = frame[:, 352:, :]

  view1 = crop_frame(view1, crop_size)
  view2 = crop_frame(view2, crop_size)
  # +1 cause the "center" frame in this case is the 0 frame.
  prev_views1 = [view1 for i in range(-window_start + 1)]
  prev_views2 = [view2 for i in range(-window_start + 1)]

  window_end = window_size + window_start - 2
  frame_num = 1
  for i in range(window_end):
    retval, frame = cap.read()

    view1 = frame[:, :352, :]
    view2 = frame[:, 352:, :]

    view1 = crop_frame(view1, crop_size)
    view2 = crop_frame(view2, crop_size)
    prev_views1.append(view1)
    prev_views2.append(view2)
    frame_num += 1

  return prev_views1, prev_views2, frame_num


def write_features(out_dir, queue):
  """Thread writer"""
  print("writer started")
  # with open("/nrs/branson/kwaki/data/c3d/debug-train.txt", "w") as fid:
  while True:
    exp_name, view1_fc, view2_fc = queue.get()
    # print(exp_name)
    if exp_name == "done":
      print("writer exiting done")
      return

    feature_file = os.path.join(out_dir, exp_name)
    # print(feature_file)
    # fid.write("%s\n" % feature_file)
    with h5py.File(feature_file, "w") as out_data:
      out_data["rgb_i3d_view1_fc"] = view1_fc
      out_data["rgb_i3d_view2_fc"] = view2_fc
  print("writer closing")


def _get_label_weight(opts, data):
  """Get number of positive examples for each label."""
  tic = time.time()
  experiments = data["exp_names"][()]
  label_mat = np.zeros((experiments.size, 7))
  vid_lengths = np.zeros((experiments.size,))
  for i in range(experiments.size):
    exp_key = experiments[i]
    exp = data["exps"][exp_key]
    for j in range(6):
      # label_counts[j] += exp["org_labels"].value[:, j].sum()
      label_mat[i, j] = exp["labels"][:, j].sum()
    # label_counts[-1] +=\
    #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()
    # label_mat[i, -1] =\
    #   exp["labels"].shape[0] - exp["labels"][()].sum()

    # vid_lengths[i] = exp["hoghof"].shape[0]
    vid_lengths[i] = exp["labels"].shape[0]

  # in this form of training, we see one negative for each positive in the
  # batch. So, just set one entry of the last column to be the number of
  # positive examples.
  label_mat[-1, -1] = label_mat[:, :6].sum()
  label_weight = 1.0 / np.mean(label_mat, axis=0)
  # label_weight = label_mat.sum(axis=0)  / np.mean(label_mat, axis=0)
  # label_weight[-2] = label_weight[-2] * 10
  if opts["flags"].reweight is False:
      label_weight = [5, 5, 5, 5, 5, 5, .01]
  print(time.time() - tic)
  return label_weight



def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  filelist = FLAGS.filelist
  # gpu_num = FLAGS.gpus
  window_size = FLAGS.window_size
  window_start = FLAGS.window_start
  batch_size = FLAGS.hantman_mini_batch
  opts = {}
  opts["flags"] = FLAGS
  opts["argv"] = sys.argv
  opts["rng"] = np.random.RandomState()
  opts["flags"].frames = list(range(window_start, window_start + window_size))

  imagenet_pretrained = FLAGS.imagenet_pretrained

  paths.setup_output_space(opts)
  g_label_names = [
    "lift", "hand", "grab", "supinate", "mouth", "chew"
  ]
  with h5py.File(opts["flags"].train_file, "r") as train_data:
    # sequences_helper.copy_templates(
    #     opts, train_data, "train", g_label_names)

    sampler = videosamplers.VideoFrameSampler(
        opts["rng"], train_data, opts["flags"].video_dir,
        opts["flags"].hantman_mini_batch,
        frames=opts["flags"].frames,
        use_pool=opts["flags"].use_pool, gpu_id=-1, normalize=crop_frame,
        max_workers=opts["flags"].max_workers,
        max_queue=opts["flags"].max_queue)

    label_weight = _get_label_weight(opts, train_data)

    NUM_CLASSES = 400
    if eval_type == 'rgb600':
      NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
      raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    # jif eval_type == 'rgb600':
    # j  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    # jelse:
    # j  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
      # RGB input has 3 channels.
      rgb_input = tf.placeholder(
        tf.float32,
        shape=(batch_size, window_size, _IMAGE_SIZE, _IMAGE_SIZE, 3))
      org_labels = tf.placeholder(
        tf.float32,
        shape=(batch_size, NUM_CLASSES)
      )
      new_labels = tf.placeholder(tf.float32, shape=(batch_size, 7))

      with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

      rgb_vars = []
      predict_var = []
      cost_var = []

      with tf.device('/gpu:0'):
        # training, use 0.5 dropout
        _, rgb_endpoints = rgb_model(
          rgb_input,
          is_training=False, dropout_keep_prob=0.5)

        rgb_logit = rgb_endpoints["logit2"]
        new_cost, optimizer = create_criterion(
          opts, rgb_logit, new_labels, weight=label_weight,
          name="hantman_crit")
      rgb_endpoints = [
        rgb_logit,
        new_cost,
        optimizer
      ]

      # remaping variables?
      rgb_variable_map = {}
      for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
          if eval_type == 'rgb600':
            rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
          else:
            # make sure not to add hantman based variable names to reload.
            if 'hantman' not in variable.name:
              rgb_variable_map[variable.name.replace(':0', '')] = variable
        # remove new_conv

      rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
      # Flow input has only 2 channels.
      rgb_input = tf.placeholder(
          tf.float32,
          shape=(batch_size, window_size, _IMAGE_SIZE, _IMAGE_SIZE, 2))
      org_labels = tf.placeholder(
        tf.float32,
        shape=(batch_size, NUM_CLASSES)
      )
      new_labels = tf.placeholder(tf.float32, shape=(batch_size, 7))

      with tf.variable_scope('Flow'):
        rgb_model = i3d.InceptionI3d(
            NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

      rgb_vars = []
      predict_var = []
      cost_var = []
      with tf.device('/gpu:0'):
        _, rgb_endpoints = rgb_model(
            rgb_input, is_training=False, dropout_keep_prob=0.5)

        rgb_logit = rgb_endpoints["logit2"]
        new_cost, optimizer = create_criterion(
          opts, rgb_logit, new_labels, weight=label_weight,
          name="hantman_crit")
      rgb_endpoints = [
        rgb_logit,
        new_cost,
        optimizer
      ]

      rgb_variable_map = {}
      for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
          # make sure not to add hantman based variable names to reload.
          if 'hantman' not in variable.name:
            rgb_variable_map[variable.name.replace(':0', '')] = variable
          # flow_variable_map[variable.name.replace(':0', '')] = variable
      rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    # movie_dir = '/nrs/branson/kwaki/data/hantman_pruned/'
    # out_dir = "/nrs/branson/kwaki/data/20180729_base_hantman/exps/i3d_64"

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {}
      if eval_type in ['flow', 'joint']:
        if imagenet_pretrained:
          rgb_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
        else:
          rgb_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
        tf.logging.info('Flow checkpoint restored')
        # flow_sample = np.load(_SAMPLE_PATHS['flow'])
        # tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
        # feed_dict[flow_input] = flow_sample

      train_network(opts, sess, rgb_saver, sampler, rgb_endpoints,
                    rgb_input, new_labels)
      print("hi")


def create_criterion(opts, endpoint, labels, weight=None, name=None):
  cost = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=endpoint,
    labels=labels
  )

  if weight is not None:
    label_weights = tf.reduce_sum(labels * weight, axis=1)
    cost = label_weights * cost

  cost = tf.reduce_mean(cost)
  if name is not None:
    optim = tf.train.GradientDescentOptimizer(
      opts["flags"].learning_rate, name=name).minimize(cost)
  else:
    optim = tf.train.GradientDescentOptimizer(
      opts["flags"].learning_rate).minimize(cost)
  return cost, optim


def augment_labels(labels):
    num_rows = labels.shape[0]
    labels = np.concatenate([labels, np.zeros((num_rows, 1))], axis=1)
    for row in range(num_rows):
        if np.all(labels[row, :] == 0):
            labels[row, -1] = 1

    return labels


def train_network(opts, sess, saver, sampler, rgb_endpoints, rgb_input, new_label):
  rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
  tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))

  timing_fid = open(
    os.path.join(opts["flags"].out_dir, "timing.csv"), "w")
  loss_fid = open(
    os.path.join(opts["flags"].out_dir, "train_loss.txt"), "w")
  count_fid = open(
    os.path.join(opts["flags"].out_dir, "counts.txt"), "w")

  # temp_fid = open(
  #   os.path.join(opts["flags"].out_dir, "debug.txt"), "w")

  h5 = h5py.File('/nrs/branson/kwaki/data/labels.hdf5', 'r')
  all_labels = h5["labels"][()]
  h5.close()
  # with('/nrs/branson/kwaki/data/labels.hdf5', "r") as h5:
  #   import pdb; pdb.set_trace()
  #   all_labels = h5["labels"]
  for i in range(opts["flags"].total_epochs):
    print("%d of %d" % (i, opts["flags"].total_epochs))
    cap = cv2.VideoCapture(
        "/nrs/branson/kwaki/data/front_flow.avi")
    if sampler.batch_idx.empty():
      sampler.reset()
    cost = 0
    epoch_tic = time.time()
    data_times = []
    network_times = []
    count = 0
    for j in range(sampler.num_batch):
      print("\t%d of %d" % (j, sampler.num_batch))
      # data_tic = time.time()
      # batch = sampler.get_minibatch()
      # # timing_fid.write("data load,%f\n" % (time.time() - data_tic))
      # data_times.append(
      #   time.time() - data_tic
      # )
      # # temp_fid.write("load,%f\n" % data_times[-1])

      data_tic = time.time()
      image_batch = np.zeros(
        (opts["flags"].hantman_mini_batch,
        opts["flags"].window_size,
        224, 224, 2), dtype="float32")
      for b in range(opts["flags"].hantman_mini_batch):
        for frame_i in range(opts["flags"].window_size):
          retval, img = cap.read()
          image_batch[b, frame_i, :, :, :] = crop_frame(img)
      data_times.append(
        time.time() - data_tic
      )

      network_tic = time.time()
      feed_dict = {}
      # feed_dict[rgb_input] = rgb_sample[:, :_SAMPLE_VIDEO_FRAMES, :, :]
      # test_label = np.zeros((1, 7)).astype('float32')
      # test_label[0][4] = 1
      # labels = batch[1]
      # labels = all_labels[i, j, :]
      labels = all_labels[0, j, :]
      labels = augment_labels(labels)

      feed_dict[new_label] = labels
      # feed_dict[rgb_input] = batch[0][:, :, :, :, :2]
      feed_dict[rgb_input] = image_batch

      out = sess.run(rgb_endpoints, feed_dict)
      predict = np.argmax(out[0], axis=1)
      gt = np.argmax(labels, axis=1)
      cost += out[1]
      network_times.append(
        time.time() - network_tic
      )
      # temp_fid.write("network,%f\n" % network_times[-1])
      count += sum(predict == gt)
      # temp_fid.flush()

    timing_fid.write("data load,%f\n" % np.mean(data_times))
    timing_fid.write("network ops,%f\n" % np.mean(network_times))
    # print(predict)
    # print(np.argmax(labels, axis=1))
    # save network
    if i % opts["flags"].save_iterations == 0:
      print("saving")
      out_path = os.path.join(
        opts["flags"].out_dir, "networks", "epoch_%04d.ckpt" % i)
      save_path = saver.save(sess, out_path)

    loss_fid.write("%f\n" % (cost / sampler.num_batch))
    loss_fid.flush()

    timing_fid.write("epoch,%f\n" % (time.time() - epoch_tic))
    timing_fid.flush()

    count_fid.write(
      "%f\n" %
      (count / (sampler.num_batch * opts["flags"].hantman_mini_batch))
    )
    count_fid.flush()
  print("saving")
  out_path = os.path.join(
    opts["flags"].out_dir, "networks", "epoch_%04d.ckpt" % i)
  save_path = saver.save(sess, out_path)
  print(predict)
  print(np.argmax(labels, axis=1))
  timing_fid.close()
  loss_fid.close()
  count_fid.close()


if __name__ == '__main__':
  tf.app.run(main)
