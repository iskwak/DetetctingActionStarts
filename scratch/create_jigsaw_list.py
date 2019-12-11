"""Script to create a list (and location) of JIGSAW Videos to process."""
from __future__ import print_function, division
# import h5py
import os
import cv2


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames


def main():
    base_dir = "/nrs/branson/kwaki/data/jigsaw_videos"
    out_dir = "/nrs/branson/kwaki/data/lists"
    out_name = os.path.join(out_dir, "jigsaw_list.txt")
    # get the suturing videos only
    video_names = os.listdir(base_dir)
    video_names.sort()

    with open(out_name, "w") as fid:
        for video_name in video_names:
            if "Suturing" in video_name:
                # to mirror the hantman_list.txt. Get frame counts
                # for the videos.
                frame_count = get_frame_count(
                    os.path.join(base_dir, video_name)
                )
                fid.write(
                    "%s,%d\n" % (video_name, frame_count)
                )

if __name__ == "__main__":
    main()
