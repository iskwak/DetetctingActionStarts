"""Helper script to combine results."""

import os
import numpy
import sys
import h5py
import helpers.paths as paths
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching
import gflags
import helpers.sequences_helper as sequences_helper

gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("display_dir", "/nrs/branson/kwaki/data/hantman_mp4", "mp4 dir")

def setup_opts(flags, output_dir):
    arg_string = [
        "create_ff_outs.py",
        "--out_dir", output_dir
    ]

    flags(arg_string)
    opts = {
        "flags": flags
    }
    opts["argv"] = arg_string

    return opts


def main(argv):
    """main"""
    FLAGS = gflags.FLAGS
    mice = ['M134', 'M147', 'M173', 'M174']
    # mouse = "M134"
    # base_dir = "/nrs/branson/kwaki/outputs/i3d_ff/adam2"
    base_dir = "/nrs/branson/kwaki/outputs/i3d_ff2/"
    input_types = ["flow", "rgb"]
    views = ["front", "side"]

    label_names = [
        "lift", "hand", "grab", "supinate", "mouth", "chew"
    ]

    base_train_name = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_%s_train.hdf5"
    base_test_name = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_%s_test.hdf5"

    for mouse in mice:
        print(mouse)
        train_name = base_train_name % mouse
        test_name = base_test_name % mouse
        # print("\ttrain")
        # with h5py.File(train_name, "r") as train_data:
        #     process_hdf5(FLAGS, train_data, base_dir, mouse, input_types, views, "train")

        print("\ttest")
        with h5py.File(test_name, "r") as test_data:
            process_hdf5(FLAGS, test_data, base_dir, mouse, input_types, views, "test")


def process_hdf5(flags, h5_data, base_dir, mouse, input_types, views, out):
    # create directory
    data_dir = os.path.join(
        "/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d",
        mouse
        )

    # setup the flow.
    print("\t\tflow")
    mouse_dir = os.path.join(base_dir, mouse, "all", input_types[0])
    opts = setup_opts(flags, mouse_dir)
    paths.setup_output_space(opts)
    compute_outputs(opts, h5_data, [input_types[0]], views, data_dir, out)

    # setup the rgb.
    print("\t\trgb")
    mouse_dir = os.path.join(base_dir, mouse, "all", input_types[1])
    opts = setup_opts(flags, mouse_dir)
    paths.setup_output_space(opts)
    compute_outputs(opts, h5_data, [input_types[1]], views, data_dir, out)

    # setup the both. in this case
    print("\t\tboth")
    mouse_dir = os.path.join(base_dir, mouse, "all", "feedforward")
    opts = setup_opts(flags, mouse_dir)
    paths.setup_output_space(opts)
    compute_outputs(opts, h5_data, input_types, views, data_dir, out)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # e_x = numpy.exp(x - numpy.max(x))
    # return e_x / e_x.sum()
    # return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)
    rows, cols = x.shape
    y = x.copy()
    for i in range(rows):
        y[i, :] = numpy.exp(y[i, :]) / numpy.sum(numpy.exp(y[i, :]))

    return y


def compute_outputs(opts, h5_data, input_types, views, base_data_dir, out_type):
    label_names = h5_data["label_names"][()]
    sequences_helper.copy_templates(
        opts, h5_data, out_type, label_names)
    # loop over the exp names
    exp_names = h5_data["exp_names"][()]

    # for each exp, get the features
    for i in range(len(exp_names)):
        # for each view
        all_logits = []
        for input_type in input_types:
            for view in views:
                feat_file = os.path.join(
                    base_data_dir, input_type, view, exp_names[i].decode())
                # print(feat_file)
                with h5py.File(feat_file, 'r') as h5_feat:
                    all_logits.append(
                        h5_feat["logits"][()]
                    )
        summed = all_logits[0]
        for j in range(1, len(all_logits)):
            summed = all_logits[1] + summed
        preds = softmax(summed)
        # now write the predictions to disk
        labels = h5_data["exps"][exp_names[i].decode()]["labels"][()]
        out_dir = os.path.join(
            opts["flags"].out_dir, "predictions", out_type
        )
        write_csvs(out_dir, exp_names[i], label_names, labels, preds)


def write_csvs(out_dir, exp_name, label_names, labels, predict):
    # frame, behavior, behavior ground truth, image
    labels = labels.reshape((labels.shape[0], 1, labels.shape[1]))
    predict = predict.reshape((predict.shape[0], 1, predict.shape[1]))
    frames = [list(range(labels.shape[0]))]
    temp = [
        label.decode() for label in label_names
    ]
    sequences_helper.write_predictions2(
        out_dir, [exp_name], predict, [labels], None, frames,
        label_names=temp)


if __name__ == "__main__":
    main(sys.argv)
