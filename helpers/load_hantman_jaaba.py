"""Load hantman jaaba features and scores."""
import scipy.io as sio
import os
import helpers.paths as paths
from sklearn.externals import joblib
import h5py
import numpy


def get_features(exp_path):
    """Given an experiment path, get the features."""
    # features are always called features.mat
    feature_filename = os.path.join(exp_path, "features.mat")
    feat_mat = sio.loadmat(feature_filename)
    features = feat_mat["curFeatures"]
    return features


def update_running_stat(n, mean, var, data):
    """Given a data array, update the running mean/std."""
    # from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        var += delta * (x - mean)
    return n, mean, var


def create_dataset(out_dir, hdf5_file, lines, all_data):
    """Create an hdf5 dataset for the hog/hoff features."""
    data_mean = 0
    data_var = 0
    data_num = 0
    count = 0
    # all_feat = []
    for line in lines:
        print(line)

        # if "M135" not in line or "2015" not in line:
        #     continue
        # if "M173" not in line:
        #     continue
        # if count > 10:
        #     break

        exp_path = paths.find_exp_dir(line)
        features = get_features(exp_path)
        for i in range(len(all_data["exp"])):
            # im dumb... the exp names are numpy arrays and need to be
            # de-ref'ed.
            if line in all_data["exp"][i][0]:
                start_idx = all_data["frame_idx"][i][0]
                # add one, the indexing is not inclusive of the end value
                end_idx = all_data["frame_idx"][i][-1] + 1

                # create a file for this experiment and then "link" it to the
                # main hdf5 file.
                out_file = os.path.join(out_dir, line)
                with h5py.File(out_file, "w") as exp_file:
                    # replace the features in the data dictionary
                    # all_data["features"][i] = features[start_idx:end_idx]
                    # update the 5 file
                    group_name = line  # + "/features"
                    group = exp_file.create_group(group_name)
                    temp = features[start_idx:end_idx]
                    # print (numpy.prod(temp.shape) * 32) / 8 / 1000
                    # all_feat.append(temp)

                    group["features"] = temp
                    group["exp_dir"] = all_data["exp_dir"][i]
                    group["frame_idx"] = all_data["frame_idx"][i]
                    group["labels"] = all_data["labels"][i]
                    # if all_data["labels"][i].shape[0] != temp.shape[0]:
                    #     import pdb; pdb.set_trace()
                    # if all_data["features"][i].shape[0] != temp.shape[0] or\
                    #         all_data["num_frames"][i] !=\
                    #         features.shape[0]:
                    #     import pdb; pdb.set_trace()
                    # group["num_frames"] = all_data["num_frames"]
                    # hdf5_file[line]["features"] =\
                    #     features[start_idx:end_idx]
                    # group = hdf5_file.create_group(group_name)
                    hdf5_file[group_name] = h5py.ExternalLink(
                        line, group_name)
                    count = count + 1
                    data_num, data_mean, data_var = update_running_stat(
                        data_num, data_mean, data_var, temp)

    # import pdb; pdb.set_trace()
    data_std = numpy.sqrt(data_var / (data_num - 1))
    # debug compare
    # all_feat = numpy.concatenate(all_feat)
    # true_mean = all_feat.mean(axis=0)
    # true_std = all_feat.std(axis=0)
    # print numpy.any(numpy.abs(true_mean - data_mean) > 0.001)
    # print numpy.any(numpy.abs(true_std - data_std) > 0.001)
    return data_mean, data_std

if __name__ == "__main__":
    used_exp_filename = "/localhome/kwaki/data/hantman/used_exps.txt"
    data_dir = "/media/drive1/data/hantman/"
    out_dir = "/media/drive1/data/hantman_processed/hoghof/"
    # out_dir =\
    #     "/media/drive1/data/hantman_processed/hoghof_single_mouse_test/"

    paths.create_dir(out_dir)
    filename = os.path.join(out_dir, "data.hdf5")

    # to help make the conversion easier, load up a previously created
    # data file.
    all_data = joblib.load((
        "/media/drive1/data/hantman_processed/"
        "joblib/test/data.npy"))
    # "joblib/relative_39window/data.npy"))

    lines = []
    with open(used_exp_filename, "r") as exp_file:
        lines = exp_file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()

    with h5py.File(filename, "w") as hdf5_file:
        hdf5_file["label_names"] = all_data["label_names"]

        exp_group = hdf5_file.create_group("exp")
        mean, std = create_dataset(out_dir, exp_group, lines, all_data)

        hdf5_file["mean"] = mean
        hdf5_file["std"] = std
