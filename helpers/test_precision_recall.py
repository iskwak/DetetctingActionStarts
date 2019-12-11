"""Analyze the output directory for a network."""
from __future__ import print_function, division
import gflags
import helpers.post_processing as post_processing
import helpers.arg_parsing as arg_parsing
import sys
import os

gflags.ADOPT_module_key_flags(arg_parsing)
gflags.DEFINE_string("input_dir", None, "Input directory path.")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(post_processing)
gflags.MarkFlagAsRequired("input_dir")

if __name__ == "__main__":
    FLAGS = gflags.FLAGS

    FLAGS(sys.argv)

    if FLAGS.help is True:
        print(FLAGS)
        exit()

    print(FLAGS.input_dir)
    thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    all_labels = []
    for i in thresholds:
        print(i)
        temp = post_processing.process_outputs(
            FLAGS.input_dir,
            "", frame_thresh=i)
        all_labels.append(temp)

    filename = os.path.join(FLAGS.input_dir, "temp.csv")
    with open(filename, "w") as out_file:
        out_file.write(
            "threshold,lift,hand,grab,supinate,mouth,chew\n"
        )
        for j in range(len(all_labels)):
            label_dicts = all_labels[j]

            out_file.write("%f" % thresholds[j])
            for i in range(0, len(label_dicts)):
                tp = float(len(label_dicts[i]['dists']))
                fp = float(label_dicts[i]['fp'])
                fn = float(label_dicts[i]['fn'])
                precision = tp / (tp + fp + 0.0001)
                recall = tp / (tp + fn + 0.0001)
                f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
                out_file.write(",%f" % f1_score)
                # print("label: %s" % label_dicts[i]['label'])
                # print("tp: %d, fp: %d, fn: %d\n" % (tp, fp, fn))
                # print("\tprecision: %f" % precision)
                # print("\trecall: %f" % recall)
                # print("\tfscore: %f" % f1_score)

            out_file.write("\n")
