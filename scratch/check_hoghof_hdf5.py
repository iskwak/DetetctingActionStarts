import h5py
import os

base_dir = '/nrs/branson/kwaki/data/hantman_hoghof/'
h5_fname = os.path.join(base_dir, 'M174_20150409_v003')
out_dir = os.path.join(base_dir, 'backup', 'cpph5')
with h5py.File(h5_fname, 'r') as h5:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    hof_side = h5["hof_front"].value
    num_frames = hof_side.shape[0]
    num_elements = hof_side.shape[1]

    for i in range(num_frames):
        # create a csv file for each
        out_name = os.path.join(out_dir, 'hof_%05d.csv' % i)
        with open(out_name, 'w') as out_fid:
            out_fid.write('%f' % hof_side[i, 0])
            for j in range(1,num_elements):
                out_fid.write(',%f' % hof_side[i, j])

#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     hog_side = h5["hog_side"].value
#     num_frames = hog_side.shape[0]
#     num_elements = hog_side.shape[1]
#
#     for i in range(num_frames):
#         # create a csv file for each
#         out_name = os.path.join(out_dir, 'hog_%05d.csv' % i)
#         with open(out_name, 'w') as out_fid:
#             out_fid.write('%f' % hog_side[i, 0])
#             for j in range(1,num_elements):
#                 out_fid.write(',%f' % hog_side[i, j])
