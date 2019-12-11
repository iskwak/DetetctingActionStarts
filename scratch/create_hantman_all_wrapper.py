"""Wrapper for setting up the hantman mice data."""
# imports are interleaved because of the GFLAGS object.

# first create the base data.
print("Creating the base setup.")
import helpers.create_base_hantman_hdf as create_base_hantman_hdf

arg_string = [
    "helpers/create_base_hantman_hdf.py",
    "--input_dir", "/media/drive1/data/hantman_processed/hdf5_data",
    # "--out_dir", "/media/drive1/data/hantman_processed/20180708_base_hantman"
    "--out_dir", "/nrs/branson/kwaki/data/20180708_base_hantman"
]

opts = create_base_hantman_hdf._setup_opts(arg_string)
create_base_hantman_hdf.main(opts)

# next create the train/test/val split
print("Creating train/test/val splits.")
import helpers.create_hantman_splits as create_hantman_splits

arg_string = [
    "helpers/create_hantman_splits.py",
    "--data", "/nrs/branson/kwaki/data/20180708_base_hantman/data.hdf5",
    "--name", "hantman",
    "--prune"
]

opts = create_hantman_splits._setup_opts(arg_string)
create_hantman_splits.create_train_test(opts)


# check the split data.
print("checking splits.")
import helpers.check_hantman_splits as check_hantman_splits
check_hantman_splits.main("/nrs/branson/kwaki/data/20180708_base_hantman/")
