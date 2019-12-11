"""Helper module to create and deal with paths."""
from __future__ import print_function, division
import os
import sys
# import helpers.paths as paths
from sklearn.externals import joblib
import helpers.git_helper as git_helper


def create_dir(path):
    """Create a path with check to see if it exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def find_exp_dir(exp_name, path="/media/drive1/data/hantman"):
    """Find the full path for the given experiment name."""
    for root, dirs, files in os.walk(path):
        if exp_name in dirs:
            # print os.path.join(root, exp_name[0])
            # return os.path.join(root, exp_name[0])
            return os.path.join(root, exp_name)
    return ""


def save_command(opts, out_dir):
    """Save the command to file."""
    with open(os.path.join(out_dir, 'command.txt'), "w") as outfile:
        for i in range(len(sys.argv)):
            outfile.write(sys.argv[i] + " ")
        outfile.write("\n")


def save_command2(out_dir, argv):
    """Save the command to file."""
    with open(os.path.join(out_dir, 'command.txt'), "w") as outfile:
        for i in range(len(argv)):
            outfile.write(argv[i] + " ")
        outfile.write("\n")


def setup_output_space(opts):
    # create the output directory
    flags = opts["flags"]
    # Wasn't able to save the GFLAGS object in a pickle easily. So clear the
    # GFLAGS, save the dictionary and on re-load just regen the flags.

    # create the output directories
    out_dir = flags.out_dir
    create_dir(out_dir)
    create_dir(out_dir + "/predictions")
    # create_dir(out_dir + "/predictions/valid")
    # create_dir(out_dir + "/predictions/train")
    # create_dir(out_dir + "/predictions/test")
    create_dir(out_dir + "/plots")
    create_dir(out_dir + "/opts")
    # create_dir(out_dir + "/grads")
    create_dir(out_dir + "/networks")

    # save the opts dict (without the gflags obj).
    opts["flags"] = None
    joblib.dump(opts, os.path.join(flags.out_dir, "opts", "opts.npy"))
    opts["flags"] = flags

    # save the command
    # save_command(opts, flags.out_dir)
    save_command2(flags.out_dir, opts["argv"])

    git_helper.log_git_status(
        os.path.join(opts["flags"].out_dir, "git_status.txt"))

    return opts
