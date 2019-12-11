"""Create the hdf data for hantman data."""
from __future__ import print_function, division
import sys
import os
import numpy
import h5py
import helpers.paths as paths
import time
import gflags
import helpers.arg_parsing as arg_parsing
from helpers.hantman_sampler import HantmanFrameGrabber
import cv2
from multiprocessing import Pool

gflags.DEFINE_string("output_dir", None, "Output directory path.")
gflags.DEFINE_string("input_dir", None, "Input data dir.")
gflags.DEFINE_string("frame_dir", None, "Frame diretory.")
gflags.ADOPT_module_key_flags(arg_parsing)

gflags.MarkFlagAsRequired("output_dir")
gflags.MarkFlagAsRequired("input_dir")
gflags.MarkFlagAsRequired("frame_dir")


def _setup_opts(argv):
    """Setup the options."""
    FLAGS = gflags.FLAGS
    opts = arg_parsing.setup_opts(argv, FLAGS)

    return opts


def _create_outputs(opts):
    """Create the output space."""
    if not os.path.exists(opts["flags"].output_dir):
        # os.makedirs(opts["flags"].out_dir)
        out_dir = os.path.join(opts["flags"].output_dir, "exps")
        os.makedirs(out_dir)


def _process_exp(input_dir, grabber, out_exp, exp_name):
    """Process experiment."""
    # exp_name = os.path.join(opts["flags"].input_dir, "exps", exp)
    # out_exp_name = os.path.join(opts["flags"].output_dir, "exps", exp)

    if "raw" in out_exp:
        del out_exp["raw"]
    if "proc" in out_exp:
        del out_exp["proc"]

    out_exp.create_group("raw")
    out_exp.create_group("proc")

    # copy the current contents.
    # import pdb; pdb.set_trace()
    # exp_filename = os.path.join(opts["flags"].input_dir, "exps", exp_name)
    exp_filename = os.path.join(input_dir, "exps", exp_name)
    # print(exp_filename)
    with h5py.File(exp_filename, "r") as exp_file:
        # the keys should be hoghof, labels, pos_features.
        out_exp["raw"]["labels"] = exp_file["labels"].value
        out_exp["proc"]["hoghof"] = exp_file["hoghof"].value
        out_exp["proc"]["pos_features"] = exp_file["pos_features"].value

    # grab the images
    grabber.set_exp(exp_name)
    frames = []
    # tic = time.time()
    for batch in grabber:
        frames.append(batch.copy())

    all_frames = numpy.concatenate(frames, axis=0)  # / 256.0
    # for i in range(all_frames.shape[0]):
    #     cv2.imshow("temp", all_frames[i, :, :, :])
    #     cv2.waitKey(10)
    # for j in range(len(frames)):
    #     print(j)
    #     temp_frames = frames[j]
    #     for i in range(temp_frames.shape[0]):
    #         cv2.imshow("temp", temp_frames[i, :, :, :])
    #         cv2.waitKey(10)
    # cv2.destroyAllWindows()
    # import pdb; pdb.set_trace()

    # split the frames into paw and side.
    side_frames = all_frames[:, :, :352, :]
    front_frames = all_frames[:, :, 352:, :]
    # print(time.time() - tic)
    # tic = time.time()
    out_exp["raw"]["img_side"] = side_frames
    out_exp["raw"]["img_front"] = front_frames
    # print(time.time() - tic)

    # parse the exp info (date/mouse).
    split_name = exp_name.split("_")
    if len(split_name) != 3:
        print("ERROR EXPERIMENT NAME")
        import pdb; pdb.set_trace()
    mouse = split_name[0]  # .encode("ascii", "ignore")
    date = split_name[1]  # .encode("ascii", "ignore")
    out_exp["mouse"] = mouse
    out_exp["date"] = date

    # for i in range(all_frames.shape[0]):
    #     cv2.imshow("img_side", side_frames[i, :, :, :])
    #     cv2.imshow("img_front", front_frames[i, :, :, :])
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()
    # import pdb; pdb.set_trace()

    return exp_name


def helper(data):
    exp = data[0]
    input_dir = data[1]
    output_dir = data[2]
    grabber = data[3]
    if "M173" not in exp:
        return("skipping")
    # out_exp_name = os.path.join(opts["flags"].output_dir, "exps", exp)
    out_exp_name = os.path.join(output_dir, "exps", exp)
    print(out_exp_name)
    with h5py.File(out_exp_name, "w") as out_exp:
        # return _process_exp(opts, grabber, out_exp, exp)
        return _process_exp(input_dir, grabber, out_exp, exp)


def main(argv):
    """Main function."""
    opts = _setup_opts(argv)
    _create_outputs(opts)

    # save the command
    paths.save_command(opts, opts["flags"].output_dir)
    paths.git_helper.log_git_status(
        os.path.join(opts["flags"].output_dir, "git_status.txt"))

    exp_path = os.path.join(opts["flags"].input_dir, "exps")
    all_exps = os.listdir(exp_path)
    all_exps.sort()

    grabber = HantmanFrameGrabber(opts["flags"].frame_dir)
    hdf_filename = os.path.join(opts["flags"].output_dir, "data.hdf5")
    tic = time.time()
    with h5py.File(hdf_filename, "w") as hdf_file:
        if "exps" in hdf_file:
            del hdf_file["exps"]
        hdf_file.create_group("exps")

        args = [
            (exp, opts["flags"].input_dir, opts["flags"].output_dir, grabber) for exp in all_exps
            # (exp, opts, grabber) for exp in all_exps
        ]

        for arg in args:
            helper(arg)

        # p = Pool(5)
        # out = p.map(helper, args[1000:1010])
        # import pdb; pdb.set_trace()
        # print("hi")
        # for exp in all_exps:
        #     if "M173" not in exp:
        #         continue
        #     print("Processing %s" % exp)
        #     out_exp_name = os.path.join(opts["flags"].output_dir, "exps", exp)
        #     with h5py.File(out_exp_name, "w") as out_exp:
        #         _process_exp(opts, grabber, out_exp, exp)

        #     hdf_file["exps"][exp] = h5py.ExternalLink(
        #         os.path.join("exps", exp), "/"
        #     )

    print(time.time() - tic)


if __name__ == "__main__":
    main(sys.argv)
