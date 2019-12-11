"""Compute means"""
import cv2
import h5py
import os
from helpers.RunningStats import RunningStats

output_dir = "/media/drive2/kwaki/data/hantman_processed/temp"
hdf_file = "/media/drive2/kwaki/data/hantman_processed/20180205/data.hdf5"

fourcc = cv2.cv.CV_FOURCC(*'XVID')


def movie_helper(out_file, frames):
    (num_frames, height, width, chan) = frames.shape
    writer = cv2.VideoWriter(
        out_file, fourcc, 30.0,
        (width, height), isColor=False)

    for frame_i in range(num_frames):
        frame = frames[frame_i, :, :, 0]
        frame = frame.astype('uint8')
        writer.write(frame)
    writer.release()

img_side_stats = RunningStats(91520)
img_front_stats = RunningStats(91520)
with h5py.File(hdf_file, "a") as h5file:
    exp_list = h5file["exps"].keys()
    exp_list.sort()

    for exp in exp_list[:100]:
        print exp
        out_file = os.path.join(output_dir, exp + "_side.avi")
        side_frames = h5file["exps"][exp]["raw"]["img_side"].value
        side_frames = side_frames[:, :, :, 0]
        side_frames = side_frames.reshape(side_frames.shape[0], -1)
        img_side_stats.add_data(side_frames)
        # movie_helper(out_file, side_frames)

        # out_file = os.path.join(output_dir, exp + "_front.avi")
        # front_frames = h5file["exps"][exp]["raw"]["img_front"].value
        # movie_helper(out_file, front_frames)

import pdb; pdb.set_trace()