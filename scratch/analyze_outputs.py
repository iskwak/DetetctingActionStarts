"""Some ideas for analyzing the sequence outputs."""
# import os
# import csv
# import numpy
# import scipy.optimize
import helpers.post_processing as post_processing
# import munkres

# post_processing.process_outputs("figs/lstm_20160819_001/predictions/valid",
#                                 "figs/lstm_20160819_001/plots")

# 39 window, reverse
# dict1 = post_processing.process_outputs(
#     "figs/lstm_20160907_003/predictions/valid",
#     "figs/lstm_20160907_003/plots")
# dict1 = post_processing.process_outputs(
#     "figs/20161213_withbalance/predictions/test",
#     "figs/20161213_withbalance/plots")
dict1 = post_processing.process_outputs(
    "/nrs/branson/kwaki/outputs/hoghofpos3/predictions/test/",
    "/nrs/branson/kwaki/outputs/hoghofpos3/plots/",
    frame_thresh=10, threshold=0.70)
#     "/nrs/branson/kwaki/outputs/hoghoff_h64/predictions/test/",
#     "/nrs/branson/kwaki/outputs/hoghoff_h64/plots/",
#     frame_thresh=10, threshold=0.70)
#     "figs/test_fast2/predictions/test/",
#     "figs/test_fast2/plots/",
#     "figs/fc_test/predictions/valid",
#     "figs/fc_test/plots")
#     "figs/test_fast2/predictions/test",
#     "figs/test_fast2/plots", frame_thresh=10)

print "feedforward"
mean_f = 0
for i in range(len(dict1)):
    tp = float(len(dict1[i]['dists']))
    fp = float(dict1[i]['fp'])
    fn = float(dict1[i]['fn'])
    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
    print "label: %s" % dict1[i]['label']
    print "\tprecision: %f" % precision
    print "\trecall: %f" % recall
    print "\tfscore: %f" % f1_score
    mean_f += f1_score
print "mean score: %f" % (mean_f / len(dict1))

print ""
print ""

dict1 = post_processing.process_outputs(
    "/nrs/branson/kwaki/outputs/hoghof_h64_new/predictions/test",
    "/nrs/branson/kwaki/outputs/hoghof_h64_new/plots",
    frame_thresh=10, threshold=0.70)
#     "figs/20161213_withoutbalance_/predictions/test",
#     "figs/20161213_withoutbalance_/plots")
#     "figs/conv_20161003_001_1hidden/predictions/valid",
#     "figs/conv_20161003_001_1hidden/plots")
#     "figs/lstm_20160930_001/predictions/valid",
#     "figs/lstm_20160930_001/plots")
#     "figs/lstm_new_error/predictions/valid",
#     "figs/lstm_new_error/plots")
# dict1 = post_processing.process_outputs(
#     "figs/conv_20160923_002/predictions/valid",
#     "figs/conv_20160923_002/plots")

print "39 forward"
mean_f = 0
for i in range(len(dict1)):
    tp = float(len(dict1[i]['dists']))
    fp = float(dict1[i]['fp'])
    fn = float(dict1[i]['fn'])
    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
    print "label: %s" % dict1[i]['label']
    print "\tprecision: %f" % precision
    print "\trecall: %f" % recall
    print "\tfscore: %f" % f1_score
    mean_f += f1_score
print "mean score: %f" % (mean_f / len(dict1))
#
# print ""
# print ""
# # print ""
# # print ""
# # print "19 reverse"
# # print "19 forward"
# #
# # dict1 = post_processing.process_outputs(
# #     "figs/lstm_20160914_003/predictions/valid",
# #     "figs/lstm_20160914_003/plots")
# #
# # mean_f = 0
# # for i in range(len(dict1)):
# #     tp = float(len(dict1[i]['dists']))
# #     fp = float(dict1[i]['fp'])
# #     fn = float(dict1[i]['fn'])
# #     precision = tp / (tp + fp)
# #     recall = tp / (tp + fn)
# #     f1_score = 2 * (precision * recall)/(precision + recall)
# #     print "label: %s" % dict1[i]['label']
# #     print "\tprecision: %f" % precision
# #     print "\trecall: %f" % recall
# #     print "\tfscore: %f" % f1_score
# #     mean_f += f1_score
# # print "mean score: %f" % (mean_f / len(dict1))
# #
# # print ""
# # print ""
# # print "moo"
# # dicts, frame_threshs = post_processing.create_precision_recall_data(
# #     "figs/lstm_20160907_001/predictions/valid")
# # post_processing.create_precision_recall_curve(
# #     "figs/lstm_20160907_001/plots", dicts, frame_threshs)
