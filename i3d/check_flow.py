import os
import sys
import gflags
import math
import sys
import h5py

gflags.DEFINE_string("filelist", "", "list of files")
gflags.DEFINE_string("feature_dir", "", "featuredir")


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
            print(exp_name)
            if os.path.exists(feature_file):
                try:
                    with h5py.File(feature_file, "r") as h5data:
                        feat1 = h5data["canned_i3d_flow_front"][()]
                        feat2 = h5data["canned_i3d_flow_side"][()]
                        print("\t%d %d" % (feat1.shape[0], feat1.shape[1]))
                        print("\t%d %d" % (feat2.shape[0], feat2.shape[1]))
                        print("\t%f" % (feat1 - feat2).mean())
                except:
                    import pdb; pdb.set_trace()
            else:
                print("\tmissing!!")
            exp_name = fid.readline().strip()


if __name__ == "__main__":
    main(sys.argv)
