import h5py
import os
import sys
import gflags

gflags.DEFINE_string("filelist", "", "list of files to compare.")
gflags.DEFINE_string("feature_dir", "", "directory of features.")


def main(argv):
    """Given a file name, loop over it and figure out whats been computed."""
    FLAGS = gflags.FLAGS
    FLAGS(argv)

    filelist = FLAGS.filelist
    feature_dir = FLAGS.feature_dir

    with open(filelist, "r") as fid:
        # loop over the lines and output the unseen files
        exp_name = fid.readline().strip()

        while exp_name:
            feature_file = os.path.join(feature_dir, exp_name)
            if not os.path.exists(feature_file):
                print(exp_name)
            exp_name = fid.readline().strip()


if __name__ == "__main__":
    main(sys.argv)