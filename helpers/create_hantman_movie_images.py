"""Create a folder of frames for each hantman movie_comb.avi."""
import cv2
import os
from . import paths

# base_path = '/localhome/kwaki/data/hantman_pruned'
# frame_out = '/localhome/kwaki/data/hantman_frames'
base_path = 'C:\\Users\ikwak\Desktop\data\hantman'
frame_out = 'C:\\Users\ikwak\Desktop\data\hantman_frames'

exp_dirs = os.listdir(base_path)
exp_dirs.sort()

for exp_dir in exp_dirs:
    print(exp_dir)
    # create a directory for the frames
    full_path = os.path.join(base_path, exp_dir)
    frame_dir = os.path.join(frame_out, exp_dir, "frames")
    paths.create_dir(frame_dir)
    movie_filename = os.path.join(full_path, "movie_comb.avi")

    # load the movie
    cap = cv2.VideoCapture(movie_filename)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frame_name = os.path.join(frame_dir, "%05d.jpg" % frame_num)
            cv2.imwrite(frame_name, frame)
            frame_num += 1
        else:
            break
    print("\t%d frames" % frame_num)

    cap.release()
