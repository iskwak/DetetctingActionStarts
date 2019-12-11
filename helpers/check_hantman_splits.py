import os
import h5py
import numpy


def check_intersection(exp_names1, exp_names2):
    intersection = numpy.intersect1d(exp_names1, exp_names2)

    return intersection.size != 0


def process_files(train_data, test_data, valid_data):
    train_exps = train_data["exp_names"].value
    test_exps = test_data["exp_names"].value
    valid_exps = valid_data["exp_names"].value

    # make sure there is no overlap
    if check_intersection(train_exps, test_exps):
        print("train and test has overlap.")
    if check_intersection(test_exps, valid_exps):
        print("test and valid has overlap.")
    if check_intersection(valid_exps, train_exps):
        print("valid and train has overlap.")


def check_filtered(skip_log):
    reasons = ["none", "early", "late", "multiple", "M135"]
    counts = [0, 0, 0, 0, 0]
    for line in skip_log:
        # get the counts for the reasons
        vals = line.rstrip().split(",")
        for i in range(len(reasons)):
            if vals[1] == reasons[i]:
                counts[i] = counts[i] + 1
    for i in range(len(reasons)):
        print("# %s: %d" % (reasons[i], counts[i]))


def main(base_dir, base_name="hantman"):
    # train_file = os.path.join(base_dir, "hantman_train.hdf5")
    # test_file = os.path.join(base_dir, "hantman_test.hdf5")
    # valid_file = os.path.join(base_dir, "hantman_valid.hdf5")
    train_file = os.path.join(base_dir, "%s_train.hdf5" % base_name)
    test_file = os.path.join(base_dir, "%s_test.hdf5" % base_name)
    valid_file = os.path.join(base_dir, "%s_valid.hdf5" % base_name)

    with h5py.File(train_file, "r") as train_data:
        with h5py.File(test_file, "r") as test_data:
            with h5py.File(valid_file, "r") as valid_data:
                process_files(train_data, test_data, valid_data)

                print("%s: %d" % (train_file, train_data["exp_names"].size))
                print("%s: %d" % (test_file, test_data["exp_names"].size))
                print("%s: %d" % (valid_file, valid_data["exp_names"].size))

    print("\n")
    # get stats on the skipped file
    # skip_logname = os.path.join(base_dir, "00_skipped.txt")
    # with open(skip_logname, "r") as skip_log:
    #     check_filtered(skip_log)


if __name__ == "__main__":
    base_dir = "/media/drive1/data/hantman_processed/20180708_base_hantman"
    main(base_dir, base_name)
