import numpy as np
import tensorflow as tf
import helpers.tf_videosamplers as videosamplers
import h5py
import sys
import helpers.paths as paths
import cv2

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
  img = img / 255 * 2 - 1
  # img = img - 1

  return img


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
    process_data(opts, sampler)


def process_data(opts, sampler):
  # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  # cap = cv2.VideoWriter(
  #   '/nrs/branson/kwaki/data/front_flow.avi', fourcc, 30.0, (224, 224))
  with h5py.File("/nrs/branson/kwaki/data/video2.hdf5", "w") as vid_dat:
    with h5py.File("/nrs/branson/kwaki/data/labels2.hdf5", "w") as h5:
      labels = np.zeros(
        (opts["flags"].total_epochs, sampler.num_batch,
        opts["flags"].hantman_mini_batch, 6),
        dtype="float32")
      count = 0
      for i in range(opts["flags"].total_epochs):
        print("%d of %d" % (i, opts["flags"].total_epochs))
        if sampler.batch_idx.empty():
          sampler.reset()

        for j in range(sampler.num_batch):
          print("\t%d of %d" % (j, sampler.num_batch))
          batch = sampler.get_minibatch()
          for b in range(len(batch[0])):
            frames = batch[0][b]
            labels[i, j, b, :] = batch[1][b]
            for frame in frames:
              frame = (frame + 1) / 2 * 255
              frame = frame.astype('uint8')
              # cap.write(frame)
            count += 1
          vid_dat["%d" % j] = ((batch[0] + 1) / 2 * 255).astype('uint8')
          # for k in range(
          # print('hi')
      h5["labels"] = labels
  # cap.release()

if __name__ == '__main__':
  tf.app.run(main)
