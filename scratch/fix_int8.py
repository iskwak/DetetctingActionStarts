"""Create movies?"""
import cv2
import h5py
import os

output_dir = "/media/drive2/kwaki/data/hantman_processed/temp"
hdf_file = "/media/drive2/kwaki/data/hantman_processed/20180205/data.hdf5"

with h5py.File(hdf_file, "a") as h5file:
    exp_list = h5file["exps"].keys()
    exp_list.sort()

    for exp in exp_list:
        print(exp)
        # out_file = os.path.join(output_dir, exp + "_side.avi")
        # import pdb; pdb.set_trace()

        side_frames = h5file["exps"][exp]["raw"]["img_side"].value
        front_frames = h5file["exps"][exp]["raw"]["img_front"].value

        side_frames = side_frames.astype("uint8")
        front_frames = front_frames.astype("uint8")

        del h5file["exps"][exp]["raw"]["img_side"]
        del h5file["exps"][exp]["raw"]["img_front"]
        h5file["exps"][exp]["raw"]["img_side"] = side_frames
        h5file["exps"][exp]["raw"]["img_front"] = front_frames
