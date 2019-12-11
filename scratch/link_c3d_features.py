"""Helper script to link the C3D features to the base hdf5 files."""
import h5py
import os
import numpy


def get_exp_names(all_exps, h5_filename):
    """Helper function to get the exp names from the hdf5 file."""
    with h5py.File(h5_filename, "r") as h5_data:
        exp_names = h5_data["exp_names"][()].astype("U")
        all_exps.append(exp_names)

def main():
    """Create external links to C3D hdf5 files."""
    # loop over each of the hdf5 files and create a global list experiments
    # to explore... Needed elsewhere anyways.
    base_dir = "/nrs/branson/kwaki/data/20180729_base_hantman"
    exp_dir = os.path.join(base_dir, "exps")
    list_dir = "/nrs/branson/kwaki/data/lists/"
    h5_filenames = [
        "hantman_split_M134_test.hdf5",
        "hantman_split_M134_train.hdf5",
        "hantman_split_M134_valid.hdf5"
    ]

    # see if the file list exists
    list_file = os.path.join(list_dir, "hantman_exp_list.txt")
    if not os.path.exists(list_file):
        # get all the exp lists.
        all_exps = []
        # loop over each hdf5 file and get the experiment names.
        for h5_filename in h5_filenames:
            h5_filename = os.path.join(base_dir, h5_filename)
            # take advantage of pass by reference?
            get_exp_names(all_exps, h5_filename)
        # merge the lists and sort
        all_exps = numpy.concatenate(all_exps)
        all_exps.sort()

        # write the files to a list.
        with open(list_file, "w") as fid:
            for exp_name in all_exps:
                fid.write("%s\n" % exp_name)
    else:
        # read the exp names.
        all_exps = []
        with open(list_file, "r") as fid:
            line = fid.readline()
            while line:
                all_exps.append(line.strip())
                line = fid.readline()

        all_exps = numpy.array(all_exps)

    # after getting the list, go through and create external links
    for exp_name in all_exps:
        print(exp_name)
        full_name = os.path.join(exp_dir, exp_name)
        with h5py.File(full_name, "a") as h5_data:
            h5_data["c3d_view1_fc6"] = h5py.ExternalLink(
                os.path.join("c3d", exp_name), "/c3d_view1_fc6"
            )
            h5_data["c3d_view2_fc6"] = h5py.ExternalLink(
                os.path.join("c3d", exp_name), "/c3d_view2_fc6"
            )
            h5_data["c3d_view1_fc7"] = h5py.ExternalLink(
                os.path.join("c3d", exp_name), "/c3d_view1_fc7"
            )
            h5_data["c3d_view2_fc7"] = h5py.ExternalLink(
                os.path.join("c3d", exp_name), "/c3d_view2_fc7"
            )


if __name__ == "__main__":
    main()