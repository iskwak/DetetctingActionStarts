"""Create movies?"""
import cv2
import h5py
import os

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
        cv2.imshow("image", frame)
        cv2.waitKey(10)
    writer.release()
    import pdb; pdb.set_trace()


with h5py.File(hdf_file, "a") as h5file:
    exp_list = h5file["exps"].keys()
    exp_list.sort()

    for exp in exp_list[:100]:
        out_file = os.path.join(output_dir, exp + "_side.avi")
        side_frames = h5file["exps"][exp]["raw"]["img_side"].value
        movie_helper(out_file, side_frames)

        out_file = os.path.join(output_dir, exp + "_front.avi")
        front_frames = h5file["exps"][exp]["raw"]["img_front"].value
        movie_helper(out_file, front_frames)


# for exp in exp_list:
#     print(exp)
#     # out_file = os.path.join(output_dir, exp + "_side.avi")
#     # import pdb; pdb.set_trace()

#     side_frames = h5file["exps"][exp]["raw"]["img_side"].value
#     front_frames = h5file["exps"][exp]["raw"]["img_front"].value

#     side_frames = side_frames.astype("uint8")
#     front_frames = front_frames.astype("uint8")

#     del h5file["exps"][exp]["raw"]["img_side"]
#     del h5file["exps"][exp]["raw"]["img_front"]
#     h5file["exps"][exp]["raw"]["img_side"] = side_frames
#     h5file["exps"][exp]["raw"]["img_front"] = front_frames
