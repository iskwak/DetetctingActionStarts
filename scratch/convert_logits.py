import h5py
import os
import numpy
import helpers.paths as paths
import gflags
import sys
import helpers.sequences_helper as sequences_helper

gflags.DEFINE_string("out_dir", "", "outdir.")
gflags.DEFINE_string("display_dir", "/nrs/branson/kwaki/data/hantman_mp4/", "display folder")


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


def main():
    h5name = "/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M174_test.hdf5"
    # feat_dir = "/nrs/branson/kwaki/data/features/finetune2_i3d/M134/rgb/side/"
    feat_dir = "/groups/branson/bransonlab/kwaki/data/features/finetune2_i3d/M174/rgb/front/"
    out_dir = "/nrs/branson/kwaki/outputs/finetune/M174/rgb/front"

    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)
    opts = {}
    opts["argv"] = sys.argv
    opts["flags"] = FLAGS
    opts["flags"].out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(opts["flags"].out_dir)
    paths.setup_output_space(opts)


    with h5py.File(h5name, "r") as h5data:
        exp_names = h5data["exp_names"][()]
        label_names = h5data['label_names'][()]
        decoded_labels = label_names.astype('unicode_')
        sequences_helper.copy_templates(
            opts, h5data, "test", label_names)
        sequences_helper.copy_templates(
            opts, h5data, "test2", label_names)
        sequences_helper.copy_templates(
            opts, h5data, "test3", label_names)
        sequences_helper.copy_templates(
            opts, h5data, "test4", label_names)

        for exp_name in exp_names:
            print(exp_name)
            feat_h5_name = os.path.join(feat_dir, exp_name.decode())
            with h5py.File(feat_h5_name, "r") as feat_data:
                logits = feat_data["logits"][()]
                probs = softmax(logits)
                labels = h5data["exps"][exp_name]["labels"][()]

                # create csvs
                r, c = probs.shape
                probs = probs.reshape((r, 1, c))
                r, c, = labels.shape
                labels = labels.reshape((r, 1, c))
                frames = [list(range(1500))]

                new_probs = probs.copy()
                # adjust labels?
                for col in range(c):
                    new_probs[:, 0, :] = probs[:, 0, :]

                exp_dir = os.path.join(out_dir, "predictions", "test")
                sequences_helper.write_predictions2(
                    exp_dir, [exp_name], new_probs[:, :, :6],
                    [labels], None, frames, label_names=decoded_labels)

                # create a max modified result
                num_rows = new_probs.shape[0]
                second_max_idx = numpy.argsort(new_probs, axis=2)[:, 0, 5].flatten()
                second_max = numpy.zeros(new_probs.shape)
                for i in range(num_rows):
                    second_max[i, 0, :] = new_probs[i, 0, second_max_idx[i]]

                new_probs_org = new_probs.copy()
                new_probs = new_probs_org - second_max
                exp_dir = os.path.join(out_dir, "predictions", "test2")
                sequences_helper.write_predictions2(
                    exp_dir, [exp_name], new_probs[:, :, :6],
                    [labels], None, frames, label_names=decoded_labels)

                new_probs = (new_probs_org - second_max) > 0
                exp_dir = os.path.join(out_dir, "predictions", "test3")
                sequences_helper.write_predictions2(
                    exp_dir, [exp_name], new_probs[:, :, :6],
                    [labels], None, frames, label_names=decoded_labels)

                new_probs = (new_probs_org - second_max) > 0.5
                exp_dir = os.path.join(out_dir, "predictions", "test4")
                sequences_helper.write_predictions2(
                    exp_dir, [exp_name], new_probs[:, :, :6],
                    [labels], None, frames, label_names=decoded_labels)


if __name__ == "__main__":
    main()
