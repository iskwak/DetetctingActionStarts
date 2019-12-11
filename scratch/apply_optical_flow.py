"""Based off of convert_hantman_data.py"""
from __future__ import print_function, division
import numpy
import numpy as np
# import argparse
import h5py
# import scipy.io as sio
# import helpers.paths as paths
# import helpers.git_helper as git_helper
import os
import time
import cv2
import cv
# import PIL


def create_movie(movie_fname, out_name, num_frames):

    cap = cv2.VideoCapture(movie_fname)
    if cap.isOpened() is False:
        import pdb; pdb.set_trace()

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    # flow_writer = cv2.VideoWriter(
    #     out_name, fourcc, 20.0,
    #     (width, height), isColor=True)

    flow = numpy.zeros((height, width, 2))
    flow_feat = numpy.zeros((num_frames, 1, height * width * 2))
    cap.set(cv.CV_CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    for frame_num in range(1, num_frames):
        # cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_num - 1)
        # ret, prev_frame = cap.read()
        cap.set(cv.CV_CAP_PROP_POS_FRAMES, frame_num)
        retval, cur_frame = cap.read()
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame[:, :, 0],  # prev img
            cur_frame[:, :, 0],   # next img
            0.5,                  # pyramid scale
            3,                    # number of levels
            15,                   # window size
            3,                    # iterations at each level
            5,                    # poly_n, polynomial expansion, recommended 5
            1.1,                  # poly_sigma, std dev for the poly expan, 1.1 rec
            cv2.OPTFLOW_USE_INITIAL_FLOW,
            flow
        )
        flow_feat[frame_num, 0, :] = flow.flatten()

        # flow_img = numpy.copy(prev_frame)
        # flow_img = draw_flow(flow_img, flow, step=8)
        # flow_writer.write(flow_img)

        prev_frame = cur_frame

    cap.release()
    # flow_writer.release()
    return flow_feat


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = numpy.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
    fx, fy = flow[y.astype('int'), x.astype('int')].T
    lines = numpy.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)

    # prune the locations with no motion.
    idx = numpy.sqrt(fx * fx + fy * fy) >= 1.25
    lines = lines[idx]
    lines = numpy.int32(lines + 0.5)

    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def main():
    # movie_file = "/media/drive1/data/hantman_pruned/M174_20150521_v014/movie_comb.avi"
    # cap = cv2.VideoCapture(movie_file)
    # cap_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    # import pdb; pdb.set_trace()
    # cap.release()
    # data_fname = "/media/drive2/data/hantman_processed/20170827_vgg/data.hdf5"
    data_fname = "/media/drive2/kwaki/data/hantman_processed/onemouse_flow2/one_mouse_multi_day_train.hdf5"
    movie_base_dir = "/media/drive2/kwaki/data/hantman_pruned/"
    with h5py.File(data_fname, "a") as hdf5_data:
        exp_names = hdf5_data["exp_names"]

        for exp_name in exp_names:
            if "M173" not in exp_name:
                continue

            print(exp_name)
            exp = hdf5_data["exps"][exp_name]
            num_frames = exp["org_labels"].shape[0]
            movie_fname = os.path.join(movie_base_dir, exp_name, "movie_comb.avi")
            out_name = os.path.join(movie_base_dir, exp_name, "flow.avi")
            tic = time.time()
            flow_feats = create_movie(movie_fname, out_name, num_frames)
            print(time.time() - tic)

            if "opt_flow" in exp.keys():
                del exp["opt_flow"]
            exp["opt_flow"] = flow_feats

    data_fname = "/media/drive2/kwaki/data/hantman_processed/onemouse_flow2/one_mouse_multi_day_test.hdf5"
    movie_base_dir = "/media/drive2/kwaki/data/hantman_pruned/"
    with h5py.File(data_fname, "a") as hdf5_data:
        exp_names = hdf5_data["exp_names"]

        for exp_name in exp_names:
            if "M173" not in exp_name:
                continue

            print(exp_name)
            exp = hdf5_data["exps"][exp_name]
            num_frames = exp["org_labels"].shape[0]
            movie_fname = os.path.join(movie_base_dir, exp_name, "movie_comb.avi")
            out_name = os.path.join(movie_base_dir, exp_name, "flow.avi")
            tic = time.time()
            flow_feats = create_movie(movie_fname, out_name, num_frames)
            print(time.time() - tic)

            if "opt_flow" in exp.keys():
                del exp["opt_flow"]
            exp["opt_flow"] = flow_feats


if __name__ == "__main__":
    main()


# arrowimg = cv2.resize(arrowimg, (0, 0), fx=2, fy=2)
# for i in range(num_rows):
#     # i is like y
#     for j in range(num_cols):
#         # j is like x
#         # flow image is (dx, dy)
#         new_y = flow[i, j, 0]
#         new_x = flow[i, j, 1]
#         delta_x = numpy.square(new_x - j)
#         delta_y = numpy.square(new_y - i)

#         if numpy.sqrt(delta_x + delta_y) >= 1:

#         # if numpy.abs(flow[i, j, 0]) >= 3 or numpy.abs(flow[i, j, 1]) >= 3:
#         #     new_pt = (new_y, new_x)
#         #     cv2.arrowedLine(
#         #         arrowimg, (j, i), new_pt, (255, 0, 0),
#         #         thickness=1
#         #     )
#         #     # import pdb; pdb.set_trace()

# arrowimg = cv2.resize(arrowimg, (0, 0), fx=0.5, fy=0.5)
# flow_writer.write(arrowimg)
# prev_frame = cur_frame
# import pdb; pdb.set_trace()