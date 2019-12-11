"""Split up the hantman data to have a smaller debug split."""
import h5py
import os

train_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_train.hdf5"
train_out_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_train.hdf5"

test_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_test.hdf5"
test_out_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_test.hdf5"

valid_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M134_valid.hdf5"
valid_out_file = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_debug_M134_valid.hdf5"

# make smaller versions of these.
def prune_hdf5(in_file, out_file):
    """Helper to prune the hdf file."""
    with h5py.File(in_file, "r") as in_data:
        with h5py.File(out_file, "w") as out_data:
            # just take the first 20
            out_data["date"] = in_data["date"][:10]
            out_data["exp_names"] = in_data["exp_names"][:10]

            out_data.create_group("exps")
            for i in range(10):
                exp_name = out_data["exp_names"][i].decode("utf-8")
                out_data["exps"][exp_name] =\
                    h5py.ExternalLink(
                        os.path.join("exps", exp_name),
                        "/"
                    )

            out_data["label_names"] = in_data["label_names"].value
            out_data["mice"] = in_data["mice"][:10]

def main():
    prune_hdf5(train_file, train_out_file)
    prune_hdf5(test_file, test_out_file)
    prune_hdf5(valid_file, valid_out_file)


if __name__ == "__main__":
    main()
