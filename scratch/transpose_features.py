# forgot to setup the external links for some hdf5 files.
import os
import h5py

# base_filename = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_train.hdf5"
out_filename = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv/one_mouse_multi_day_train.hdf5"
# base_filename = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_test.hdf5"
# out_filename = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv/one_mouse_multi_day_test.hdf5"


with h5py.File(out_filename, "a") as out_file:
    exp_list = out_file["exp_names"]

    # for each experiment in the list create an external link.
    for exp_name in exp_list:
        print(exp_name)
        data = out_file["exps"][exp_name]["conv3d"].value
        del out_file["exps"][exp_name]["conv3d"]
        data = data.reshape(data.shape[0], 1, data.shape[1])
        out_file["exps"][exp_name]["conv3d"] = data


out_filename = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv/one_mouse_multi_day_test.hdf5"

with h5py.File(out_filename, "a") as out_file:
    exp_list = out_file["exp_names"]

    # for each experiment in the list create an external link.
    for exp_name in exp_list:
        print(exp_name)
        data = out_file["exps"][exp_name]["conv3d"].value
        del out_file["exps"][exp_name]["conv3d"]
        data = data.reshape(data.shape[0], 1, data.shape[1])
        out_file["exps"][exp_name]["conv3d"] = data
