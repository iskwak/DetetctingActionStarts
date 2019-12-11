"""Analyze the output directory for a network."""
from __future__ import print_function, division
import gflags
import helpers.post_processing as post_processing
import helpers.arg_parsing as arg_parsing
import sys

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
    labels = [
        'lift', 'hand', 'grab', 'supinate', 'mouth', 'chew'
    ]
    label_dicts = post_processing.process_outputs(
        FLAGS.input_dir, "", labels)

    mean_f = 0
    for i in range(len(label_dicts)):
        tp = float(len(label_dicts[i]['dists']))
        fp = float(label_dicts[i]['fp'])
        fn = float(label_dicts[i]['fn'])
        precision = tp / (tp + fp + 0.0001)
        recall = tp / (tp + fn + 0.0001)
        f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
        print("label: %s" % label_dicts[i]['label'])
        print("tp: %d, fp: %d, fn: %d\n" % (tp, fp, fn))
        print("\tprecision: %f" % precision)
        print("\trecall: %f" % recall)
        print("\tfscore: %f" % f1_score)
        mean_f += f1_score
    print("mean score: %f" % (mean_f / len(label_dicts)))
