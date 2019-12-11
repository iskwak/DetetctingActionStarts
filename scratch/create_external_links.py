# forgot to setup the external links for some hdf5 files.
import os
import h5py

# base_filename = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_train.hdf5"
# out_filename = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv/one_mouse_multi_day_train.hdf5"
base_filename = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_test.hdf5"
out_filename = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv/one_mouse_multi_day_test.hdf5"

with h5py.File(base_filename, "r") as in_file:
    with h5py.File(out_filename, "a") as out_file:
        exp_list = in_file["exp_names"]

        # for each experiment in the list create an external link.
        for exp_name in exp_list:
            print(exp_name)
            # import pdb; pdb.set_trace()
            out_file["exps"][exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name), "/"
            )
