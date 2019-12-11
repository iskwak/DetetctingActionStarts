"""Create train/test splits for jigsaw."""
import os
import sys
import h5py
import numpy
import re


def get_experiments(filename):
    exp_names = []
    subjects = []
    with open(filename, "r") as fd:
        for line in fd:
            # each line is <experiment>_<start frame>_<end_frame>.txt<some number of spaces><label>
            # only care about the experiment part. so the first token.
            exp_info = line.split(" ")[0]
            # extract the experiment name
            # everything before the 3rd underscore
            if "Suturing" in exp_info:
                exp_name = re.findall("^([^_]*_[^_]*)", exp_info)[0]
                subject = re.findall("^[^_]*_([^_]*)", exp_info)[0]
            else:
                exp_name = re.findall("^([^_]*_[^_]*_[^_]*)", exp_info)[0]
                subject = re.findall("^[^_]*_[^_]*_([^_]*)", exp_info)[0]

            exp_names.append(exp_name)
            subject = subject[0]
            subjects.append(subject)

    exp_names = numpy.unique(exp_names)
    subjects = numpy.unique(subjects)
    return exp_names, subjects


def create_hdfs(hdf_name, exps, subjects):
    print(hdf_name)
    with h5py.File(hdf_name, "w") as hdf5:
        hdf5.create_group("exps")
        hdf5["exp_names"] = exps
        hdf5["subjects"] = subjects

        for exp in exps:
            hdf5["exps"][exp] = h5py.ExternalLink(
                os.path.join("exps", exp), "/"
            )


def process_split(base_name, hdf_dir, train_file, test_file):
    # for the training and test file, create a train and test hdf file.
    # initially, just have all the split sperate...?
    # Don't worry about user1_out is the same for each task just yet.

    # get training/testing experiment names
    train_exps, train_subj = get_experiments(train_file)
    test_exps, test_subj = get_experiments(test_file)

    # create a hdf5 file for each
    outname = os.path.join(hdf_dir, "%s_train.hdf5" % base_name)
    create_hdfs(outname, train_exps, train_subj)

    outname = os.path.join(hdf_dir, "%s_test.hdf5" % base_name)
    create_hdfs(outname, test_exps, test_subj)


def main():
    split_base = "/media/drive3/kwaki/data/jigsaw/original/Experimental_setup"
    hdf_dir = "/media/drive3/kwaki/data/jigsaw/20180619_jigsaw_base/"
    base_hdf = os.path.join(hdf_dir, "data.hdf5")
    sub_dir = "unBalanced/GestureClassification/UserOut"

    # for each task, get the unbalanced user out trials.
    tasks = os.listdir(split_base)
    tasks.sort()
    for i in range(len(tasks)):
        task_path = os.path.join(split_base, tasks[i], sub_dir)
        user_splits = os.listdir(task_path)
        user_splits.sort()

        for j in range(len(user_splits)):
            base_txt_dir = os.path.join(task_path, user_splits[j], "itr_1")
            test_file = os.path.join(base_txt_dir, "Test.txt")
            train_file = os.path.join(base_txt_dir, "Train.txt")
            # construct the name of the split, based off task and user out.
            base_name = "%s_%s" % (tasks[i], user_splits[j])
            process_split(base_name, hdf_dir, train_file, test_file)

    # create a test dataset.
    # just use on test set as traing and test.
    task = tasks[2]
    task_path = os.path.join(split_base, task, sub_dir)
    user_splits = os.listdir(task_path)
    user_splits.sort()
    user_split = user_splits[0]

    base_txt_dir = os.path.join(task_path, user_split, "itr_1")
    test_file = os.path.join(base_txt_dir, "Test.txt")
    # construct the name of the split, based off task and user out.
    base_name = "debug_%s_%s" % (task, user_split)
    # use test file for train and test.
    process_split(base_name, hdf_dir, test_file, test_file)


if __name__ == "__main__":
    main()
