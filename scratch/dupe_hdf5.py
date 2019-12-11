"""Duplicate the hdf5 files."""
import h5py
import os


input_file = "/media/drive1/data/hantman_processed/20170827_vgg/one_mouse_multi_day_train.hdf5"
output_file = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_train.hdf5"
# input_file = "/media/drive1/data/hantman_processed/20170827_vgg/one_mouse_multi_day_test.hdf5"
# output_file = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_test.hdf5"

with h5py.File(input_file, "r") as in_file:
    with h5py.File(output_file, "w") as out_file:
        # import pdb; pdb.set_trace()
        exp_list = in_file["exp_names"].value
        # exp_list = exp_list.sort()
        exp_list.sort()

        out_file["date"] = in_file["date"].value
        out_file["exp_names"] = exp_list
        out_file["mice"] = in_file["mice"].value

        out_file.create_group("exps")
        for exp in exp_list:
            print(exp)
            out_file["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/"
            )
