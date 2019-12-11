import h5py
import os
import numpy
import cv2


def main():
    base_dir = "/groups/branson/bransonlab/kwaki/data/thumos14/"
    exp_dir = os.path.join(base_dir, "h5data", "exps")
    labels_fname = os.path.join(base_dir, "meta", "labels.txt")

    label_names = []
    with open(labels_fname, "r") as fid:
        name = fid.readline().strip()
        while name:
            label_names.append(name)
            name = fid.readline().strip()

    # get the fps's...
    fps_dict = {}
    with open(os.path.join(base_dir, "meta", "fps.txt"), "r") as fid:
        line = fid.readline().strip()
        while line:
            name, fps = line.split(', ')
            fps_dict[name] = float(fps)
            line = fid.readline().strip()

    # loop over the videos
    valid_videos = []
    test_videos = []
    for i in range(len(label_names)):
        # create the data
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'valid', 'annotation', '%s_val.txt' % label_names[i])
        videos = get_videos(base_dir, exp_dir, i, label_annotation_file, fps_dict, label_names[i])
        valid_videos = valid_videos + videos

        # also do test files
        label_annotation_file = os.path.join(
            base_dir, 'meta', 'test', 'annotation', '%s_test.txt' % label_names[i])
        videos = get_videos(base_dir, exp_dir, i, label_annotation_file, fps_dict, label_names[i])
        test_videos = test_videos + videos

    # create the meta files for train/test
    # valid_filename = os.path.join(base_dir, "h5data", "train.hdf5")
    # create_meta(valid_filename, label_names, valid_videos)
    # test_filename = os.path.join(base_dir, "h5data", "test.hdf5")
    # create_meta(test_filename, label_names, test_videos)


if __name__ == "__main__":
    main()

