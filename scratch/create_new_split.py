import os
import h5py
import sys
import numpy


def get_experiments(h5name):
    exp_names = []
    exp_dates = []
    data = {}
    with h5py.File(h5name, "r") as h5data:
        data['date'] = h5data["date"][()]
        data['exp_names'] = h5data['exp_names'][()]
        data['exps'] = None
        data['label_names'] = h5data['label_names'][()]
        data['mice'] = h5data['mice'][()]

    return data


def split_dates(exps, dates):
    total_exps = exps.shape[0]
    half_exp = 1.0 * total_exps / 2
    unique_dates = numpy.unique(dates)
    total_dates = unique_dates.shape[0]

    # want roughly half in train/test.
    # prob want more in test than train. so start from
    # the last date and work backwards.
    num_exps = 0
    train_dates = unique_dates.tolist()
    test_dates = []
    for date in numpy.flip(unique_dates):
        num_exps += (dates == date).sum()
        test_dates.append(date)
        train_dates = train_dates[:-1]
        if num_exps > half_exp:
            break
    print("\t%d %d" % (num_exps, total_exps - num_exps))
    return numpy.asarray(train_dates), numpy.asarray(test_dates)


def sanity_check(train, test, all_dates):
    if (train.shape[0] + test.shape[0]) != all_dates.shape[0]:
        print("failed lengths")
        import pdb; pdb.set_trace()
        return False
    if numpy.intersect1d(train, test).size != 0:
        print("intersect failed")
        import pdb; pdb.set_trace()
        return False
    return True


def create_splits(train_mouse, base_dir, exp_dict):
    # split the dates in "half"
    exps = exp_dict[train_mouse]['exp_names']
    dates = exp_dict[train_mouse]['date']
    train, test = split_dates(exps, dates)

    # some sanity checking...
    sanity_check(train, test, numpy.unique(dates))

    # actually create the new HDF5 files.
    base_out_name = "hantman_split_half_%s_%s.hdf5"
    # test hdf5 is easy, start there
    test_name = os.path.join(
        base_dir, base_out_name % (train_mouse, 'test'))

    test_mask = create_date_mask(dates, test)
    with h5py.File(test_name, "w") as h5data:
        h5data["label_names"] = exp_dict[train_mouse]["label_names"]
        h5data["date"] = exp_dict[train_mouse]["date"][test_mask]
        h5data["exp_names"] = exp_dict[train_mouse]["exp_names"][test_mask]
        h5data["mice"] = exp_dict[train_mouse]["mice"][test_mask]

        h5data.create_group("exps")
        for exp_name in exp_dict[train_mouse]["exp_names"][test_mask]:
            h5data["exps"][exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name), "/")

    # next create the train split.
    # this includes things from the train mouse, plus the rest of the
    # mice.
    all_mice = exp_dict.keys()
    all_mice.sort()
    date = numpy.array([])
    exp_names = numpy.array([])
    mice = numpy.array([])
    for mouse in all_mice:
        if mouse == train_mouse:
            train_mask = create_date_mask(dates, train)
            cur_date = exp_dict[mouse]["date"][train_mask]
            cur_names = exp_dict[mouse]["exp_names"][train_mask]
            cur_mice = exp_dict[mouse]["mice"][train_mask]
        else:
            cur_date = exp_dict[mouse]["date"][()]
            cur_names = exp_dict[mouse]["exp_names"][()]
            cur_mice = exp_dict[mouse]["mice"][()]
        date = numpy.concatenate([date, cur_date])
        exp_names = numpy.concatenate([exp_names, cur_names])
        mice = numpy.concatenate([mice, cur_mice])

    train_name = os.path.join(
        base_dir, base_out_name % (train_mouse, 'train'))
    with h5py.File(train_name, "w") as h5data:
        h5data["label_names"] = exp_dict[train_mouse]["label_names"]
        h5data["date"] = date
        h5data["exp_names"] = exp_names
        h5data["mice"] = mice

        h5data.create_group("exps")
        for exp_name in exp_names:
            h5data["exps"][exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name), "/")


def create_date_mask(dates, sub_dates):
    mask = dates == 'spam'
    for mask_date in sub_dates:
        mask += dates == mask_date
    return mask


def main(argv):
    # load up the h5py files.
    base_dir = '/nrs/branson/kwaki/data/20180729_base_hantman'

    # the test splits are the videos for each mouse
    h5_names = [
        "hantman_split_M134_test.hdf5",
        "hantman_split_M147_test.hdf5",
        "hantman_split_M173_test.hdf5",
        "hantman_split_M174_test.hdf5"
    ]
    print(h5_names[0])
    M134_name = os.path.join(base_dir, h5_names[0])
    M134_data = get_experiments(M134_name)

    print(h5_names[1])
    M147_name = os.path.join(base_dir, h5_names[1])
    M147_data = get_experiments(M147_name)

    print(h5_names[2])
    M173_name = os.path.join(base_dir, h5_names[2])
    M173_data = get_experiments(M173_name)

    print(h5_names[3])
    M174_name = os.path.join(base_dir, h5_names[3])
    M174_data = get_experiments(M174_name)

    exp_dict = {
        'M134': M134_data,
        'M147': M147_data,
        'M173': M173_data,
        'M174': M174_data
    }

    create_splits('M134', base_dir, exp_dict)
    create_splits('M147', base_dir, exp_dict)
    create_splits('M173', base_dir, exp_dict)
    create_splits('M174', base_dir, exp_dict)


if __name__ == "__main__":
    main(sys.argv)