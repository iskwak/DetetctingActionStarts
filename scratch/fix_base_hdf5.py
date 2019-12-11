"""Create 3d conv features."""
import sys
import os
import h5py
# from helpers.RunningStats import RunningStats
import helpers.paths as paths
import helpers.git_helper as git_helper
import time
# import torch.nn.Parameter as Parameter

input_name = "/media/drive2/kwaki/data/hantman_processed/20180206/one_mouse_multi_day_train.hdf5"
out_dir = "/media/drive2/kwaki/data/hantman_processed/20180212_3dconv"


def main():
    output_name = os.path.join(out_dir, "one_mouse_multi_day_train.hdf5")

    with h5py.File(input_name, "r") as in_file:
        with h5py.File(output_name, "w") as out_file:
            # copy over the general stats
            print("a")
            out_file["date"] = in_file["date"].value
            out_file["exp_names"] = in_file["exp_names"].value
            out_file["mice"] = in_file["mice"].value
            print("b")
            out_file.create_group("exps")
            for exp_name in in_file["exp_names"]:
                print(exp_name)
                out_file["exps"][exp_name] = h5py.ExternalLink(
                    os.path.join("exps", exp_name), "/"
                )
            #     tic = time.time()
            #     process_exp(exp_name, in_file, out_dir)
            #     print("\t%f" % (time.time() - tic))


if __name__ == "__main__":
    # paths.create_dir(out_dir)
    # paths.create_dir(os.path.join(out_dir, "exps"))
    temp_dirname = os.path.join(out_dir, "add_pos_features")
    paths.create_dir(temp_dirname)
    paths.save_command2(temp_dirname, sys.argv)
    git_helper.log_git_status(
        os.path.join(temp_dirname, "00_git_status.txt"))

    main()
