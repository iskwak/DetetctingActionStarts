# setup symlinks for the gpuflow code to use.
# gpuflow code expects all the videos to be in one folder.
import os

input_dir = "/nrs/branson/kwaki/data/hantman_pruned"
output_dir = "/nrs/branson/kwaki/data/hantman_vid_sym"

input_list = "/nrs/branson/kwaki/data/lists/hantman_exp_list.txt"

with open(input_list, "r") as fid:
    # loop over the lines
    for line in fid:
        exp_name = line.strip()
        # construct the original movie file name
        video_name = os.path.join(input_dir, exp_name, "movie_comb.avi")
        # construct the output symlink name
        sym_name = os.path.join(output_dir, "%s.avi" % exp_name)
        # print(sym_name)
        os.symlink(video_name, sym_name)
