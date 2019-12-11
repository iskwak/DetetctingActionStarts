import os
import sys
import gflags
import math

gflags.DEFINE_string("filelist", "", "list of files")
gflags.DEFINE_integer("splits", 0, "number of splits")
gflags.DEFINE_string("baseout", "", "output directory/base filename")

def main(argv):
    """create split files for processing"""
    flags = gflags.FLAGS
    flags(argv)
    splits = flags.splits
    baseout = flags.baseout

    filelist = flags.filelist
    filenames = []
    with open(filelist, "r") as fid:
        exp_name = fid.readline().strip()

        while exp_name:
            filenames.append(exp_name)
            exp_name = fid.readline().strip()

    num_files = len(filenames)
    base_count = math.floor(num_files * 1.0 / splits)
    rem = num_files % splits

    file_idx = 0
    for split_i in range(splits):
        # loop over the splits.
        # add base_count files to the new file
        out_name = "%s_%d.txt" % (baseout, split_i)
        with open(out_name, "w") as out_fid:
            for count_i in range(base_count):
                out_fid.write("%s\n" % filenames[file_idx])
                file_idx += 1
            if split_i < rem:
                out_fid.write("%s\n" % filenames[file_idx])
                file_idx += 1


if __name__ == "__main__":
    main(sys.argv)