"""Convert matlab mouse data into a pickle."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from helpers.RunningStats import RunningStats
from sklearn.externals import joblib
import os


def get_exp(date_idx, mouse_idx, exps, data):
    sampled = []
    for i in range(len(exps)):
        if (mouse_idx[i] == True) and (date_idx[i] == True):
            # return exps[i]
            sampled.append(exps[i])
            print(data["exps"][exps[i]]["labels"][:1500, :].sum(axis=0))
            if len(sampled) == 2:
                return sampled


def get_examples(mice, exps, dates, data):
    unique_mice = numpy.unique(mice)

    sampled = []
    for mouse in unique_mice:
        mouse_idx = mice == mouse
        mouse_dates = dates[mouse_idx]
        unique_dates = numpy.unique(mouse_dates)

        # use first 3 dates?
        date_idx = unique_dates[0] == dates
        sampled += get_exp(date_idx, mouse_idx, exps, data)
        # sampled.append(
        #     get_exp(date_idx, mouse_idx, exps))

        # date_idx = unique_dates[1] == dates
        # sampled += get_exp(date_idx, mouse_idx, exps, data)
        # sampled.append(
        #     get_exp(date_idx, mouse_idx, exps))

        date_idx = unique_dates[2] == dates
        sampled += get_exp(date_idx, mouse_idx, exps, data)
        # sampled.append(
        #     get_exp(date_idx, mouse_idx, exps))
    return sampled


def main():
    fname = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5"
    # fname = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5"
    out_dir = "/nrs/branson/kwaki/data/20180729_base_hantman/exps/pca_i3d"

    ipcas = {
        'rgb_i3d_view1_fc_64': IncrementalPCA(n_components=2000, batch_size=400),
        'rgb_i3d_view2_fc_64': IncrementalPCA(n_components=2000, batch_size=400),
        'flow_i3d_view1_fc': IncrementalPCA(n_components=2000, batch_size=400),
        'flow_i3d_view2_fc': IncrementalPCA(n_components=2000, batch_size=400),
    }
    keys = ipcas.keys()
    # keys = ['rgb_i3d_view1_fc_64', 'flow_i3d_view1_fc']

    load = True
    if load is True:
        out_name = os.path.join(out_dir, "stats")
        stats_data = joblib.load(out_name)
        ipcas = stats_data[0]
        stats = stats_data[1]
    else:
        # first get the mean and variance
        with h5py.File(fname, "r") as h5_data:
            exps = h5_data["exp_names"][()]
            mice = h5_data["mice"][()]
            date = h5_data["date"][()]
            sampled = get_examples(mice, exps, date, h5_data)
            # sampled = sampled[:2]
            # print(sampled)
            # temp = sampled
            # sampled = sampled[:-1:6]

            stats = {
                'rgb_i3d_view1_fc_64': RunningStats(7168),
                'rgb_i3d_view2_fc_64': RunningStats(7168),
                'flow_i3d_view1_fc': RunningStats(7168),
                'flow_i3d_view2_fc': RunningStats(7168),
            }

            for exp in sampled:
                print exp
                tic = time.time()
                max_idx = numpy.min([
                    h5_data["exps"][exp]["rgb_i3d_view1_fc_64"].shape[0],
                    1500])
                for key in keys:
                    exp_data = h5_data["exps"][exp][key][:max_idx, :]
                    stats[key].add_data(exp_data)
                print "\t%f" % (time.time() - tic)

        # run the IPCA algorithm
        # only on the sampled data
        print(sampled)
        with h5py.File(fname, "r") as h5_data:
            tic = time.time()
            total_tic = time.time()
            print("partialfitting")
            print("%d" % len(sampled))
            for key in keys:
                print("\t%s" % key)
                data_mean = stats[key].mean
                data_std = stats[key].compute_std()
                temp_data = []
                for exp in sampled:
                    print("\t%s" % exp)
                    max_idx = numpy.min([
                        h5_data["exps"][exp][key].shape[0],
                        1500])

                    exp_data= h5_data["exps"][exp][key][:max_idx, :]
                    temp_data.append((exp_data - data_mean) / data_std)

                    # num rows should be the same in side and front
                    num_rows = sum([data.shape[0] for data in temp_data])
                    if num_rows > 2000:
                        tic = time.time()
                        # once the number of rows is greater than the number of
                        # components run the ipca algo
                        full_data = numpy.concatenate(temp_data)

                        ipcas[key].partial_fit(full_data)
                        print "\t%f" % (time.time() - tic)
                        # reset the temp_data list
                        temp_data = []
            print "total tic: %f" % (time.time() - total_tic)

        print('saving')
        out_name = os.path.join(out_dir, "stats")
        joblib.dump(
            [ipcas, stats], out_name)

    # next apply ipca
    with h5py.File(fname, "r") as h5_data:
        exps = h5_data["exp_names"][()]
        for exp in exps:
            print(exp)
            out_exp = os.path.join(out_dir, exp)
            max_idx = numpy.min([
                h5_data["exps"][exp]['rgb_i3d_view1_fc_64'].shape[0],
                1500])
            with h5py.File(out_exp, "w") as new_exp:
                features = []
                tic = time.time()
                for key in keys:
                    data_mean = stats[key].mean
                    data_std = stats[key].compute_std()

                    reduced = (
                        (h5_data["exps"][exp][key][:max_idx, :] - data_mean) /
                        data_std)
                    reduced = numpy.dot(
                        reduced - ipcas[key].mean_,
                        ipcas[key].components_[:2000, :].T)
                    features.append(reduced)
                print(time.time() - tic)
                all_feat = numpy.concatenate(features, axis=1)
                new_exp['reduced_i3d'] = all_feat


if __name__ == "__main__":
    main()
