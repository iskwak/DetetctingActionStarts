"""Duplicate the hdf5 files."""
from __future__ import print_function, division
import h5py
import os
import numpy


base_dir = "/nrs/branson/kwaki/data/20180212_3dconv/"
train_file = "/nrs/branson/kwaki/data/20180212_3dconv/one_mouse_multi_day_train.hdf5"
test_file = "/nrs/branson/kwaki/data/20180212_3dconv/one_mouse_multi_day_test.hdf5"
# input_file = "/media/drive1/data/hantman_processed/20170827_vgg/one_mouse_multi_day_test.hdf5"
# output_file = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_test.hdf5"


# first get the all the exp names
def get_exps_data(h5_data):
    dates = h5_data["date"].value
    exp_names = h5_data["exp_names"].value
    mice = h5_data["mice"].value
    # import pdb; pdb.set_trace()
    return dates, exp_names, mice


def create_h5(out_name, dates, exps, mice):
    with h5py.File(out_name, "w") as h5_data:
        h5_data["date"] = dates
        h5_data["exp_names"] = exps
        h5_data["mice"] = exps
        h5_data.create_group("exps")

        for exp_name in exps:
            h5_data["exps"][exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name), "/"
            )


def main():
    rng = numpy.random.RandomState(123)
    with h5py.File(train_file, "r") as h5_data:
        train_dates, train_exps, train_mice = get_exps_data(h5_data)
    with h5py.File(test_file, "r") as h5_data:
        test_dates, test_exps, test_mice = get_exps_data(h5_data)

    # merge the data
    dates = numpy.concatenate([train_dates, test_dates])
    exp_names = numpy.concatenate([train_exps, test_exps])
    mice = numpy.concatenate([train_mice, test_mice])

    # randomly sample the videos to make a 70/15/15 split for train/val
    num_vids = len(dates)
    unique_dates = numpy.unique(dates)

    rand_idx = rng.permutation(len(unique_dates))
    # split_idx = int(numpy.floor(num_vids * 0.7))
    # train_idx = rand_idx[:split_idx]
    # test_idx = rand_idx[split_idx:]
    train_dates = numpy.array([])
    train_exps = numpy.array([])
    train_mice = numpy.array([])
    for i in range(len(rand_idx)):
        if len(train_dates) > (num_vids * 0.7):
            prev_idx = i
            break
        cur_date = unique_dates[rand_idx[i]]
        date_idx = cur_date == dates
        train_dates = numpy.concatenate([train_dates, dates[date_idx]])
        train_exps = numpy.concatenate([train_exps, exp_names[date_idx]])
        train_mice = numpy.concatenate([train_mice, exp_names[date_idx]])

    # import pdb; pdb.set_trace()
    val_dates = numpy.array([])
    val_exps = numpy.array([])
    val_mice = numpy.array([])
    for i in range(prev_idx, len(rand_idx)):
        if len(val_dates) > (num_vids * 0.15):
            prev_idx = i
            break
        cur_date = unique_dates[rand_idx[i]]
        date_idx = cur_date == dates
        val_dates = numpy.concatenate([val_dates, dates[date_idx]])
        val_exps = numpy.concatenate([val_exps, exp_names[date_idx]])
        val_mice = numpy.concatenate([val_mice, exp_names[date_idx]])

    test_dates = numpy.array([])
    test_exps = numpy.array([])
    test_mice = numpy.array([])
    for i in range(prev_idx, len(rand_idx)):
        cur_date = unique_dates[rand_idx[i]]
        date_idx = cur_date == dates
        test_dates = numpy.concatenate([test_dates, dates[date_idx]])
        test_exps = numpy.concatenate([test_exps, exp_names[date_idx]])
        test_mice = numpy.concatenate([test_mice, exp_names[date_idx]])

    # create the hdfs
    out_name = os.path.join(base_dir, "m173_train.hdf5")
    create_h5(out_name, train_dates, train_exps, train_mice)

    out_name = os.path.join(base_dir, "m173_val.hdf5")
    create_h5(out_name, val_dates, val_exps, val_mice)

    out_name = os.path.join(base_dir, "m173_test.hdf5")
    create_h5(out_name, test_dates, test_exps, test_mice)

    print("hi")


if __name__ == "__main__":
    main()
