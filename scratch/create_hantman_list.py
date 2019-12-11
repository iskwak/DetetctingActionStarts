"""Helper script to create a list of files and the number of frames to process."""
import h5py


def process_hdf(fid, hdf_name):
    # loop over the experiments in the hdf5, get the experiment names, and
    # the number of frames. This way the c++ won't have to go through all the
    # frames in each of the videos.
    with h5py.File(hdf_name, "r") as data:
        exp_names = data["exp_names"].value
        exp_names.sort()

        for exp_name in exp_names:
            num_labels = data["exps"][exp_name]["labels"].value.shape[0]
            # print("%s,%s\n" % (exp_name, num_labels))
            fid.write("%s,%s\n" % (exp_name, num_labels))


def main():
    # input_files = [
    #     '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_test.hdf5',
    #     '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_train.hdf5',
    #     '/nrs/branson/kwaki/data/20180708_base_hantman/hantman_valid.hdf5'
    # ]
    input_files = [
        "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_train.hdf5",
        "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_valid.hdf5",
        "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_M174_test.hdf5"
    ]

    with open("/nrs/branson/kwaki/data/hantman_list2.txt", "w") as fid:
        for hdf_name in input_files:
            process_hdf(fid, hdf_name)


if __name__ == "__main__":
    main()
