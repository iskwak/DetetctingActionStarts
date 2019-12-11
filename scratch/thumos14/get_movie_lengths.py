import h5py
import os
import numpy

dataname = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5"

train_num_frames = []
with h5py.File(dataname, "r") as h5data:
    exp_names = h5data["exp_names"]

    for i in range(len(exp_names)):
        train_num_frames.append(
            h5data["exps"][exp_names[i]]["labels"].shape[0]
        )

dataname = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/test.hdf5"
test_num_frames = []
with h5py.File(dataname, "r") as h5data:
    exp_names = h5data["exp_names"]

    for i in range(len(exp_names)):
        test_num_frames.append(
            h5data["exps"][exp_names[i]]["labels"].shape[0]
        )

train_counts, train_bins = numpy.histogram(numpy.asarray(test_num_frames))

# convert bins to "x" coordinates
train_xs = []
for i in range(len(train_bins) - 1):
    train_xs.append(
        (train_bins[i + 1] - train_bins[i]) / 2 + train_bins[i]
    )

with open("/nrs/branson/kwaki/outputs/analysis/thumos_stats/data.csv", "w") as fid:
    fid.write("x, counts\n")
    for i in range(len(train_xs)):
        fid.write("%f,%f\n" % (train_xs[i], train_counts[i]))


# with open("/nrs/branson/kwaki/outputs/analysis/thumos_stats/train.csv", "w") as fid:
#     fid.write("x, counts\n")
#     for i in range(len(xs)):
#         fid.write("%f,%f\n" % (xs[i], counts[i]))



# import pdb;pdb.set_trace()
