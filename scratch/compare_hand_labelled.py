"""Compare hand labelled outputs."""
import helpers.post_processing as post_processing
import numpy

dict1 = post_processing.process_outputs("figs/labels",
                                        "figs/labels/plots")
dict2 = post_processing.process_outputs("figs/labels2",
                                        "figs/labels2/plots")

# for i in range(len(dict1)):
#     print "label: %s" % dict1[i]['label']
#     print "\ttotal: %d" % dict1[i]['total']
#     print "\tmean 1: %f" % numpy.mean(dict1[i]['dists'])
#     print "\tmean 2: %f" % numpy.mean(dict2[i]['dists'])
#     print "\tstd 1: %f" % numpy.std(dict1[i]['dists'])
#     print "\tstd 2: %f" % numpy.std(dict2[i]['dists'])
#     print "\tfn 1: %f" % dict1[i]['fn']
#     print "\tfn 2: %f" % dict2[i]['fn']

# compare the hand labelled outputs.
run1 = "figs/labels/"
run2 = "figs/labels2/"
exp1 = "figs/labels/M134_20141204_v009/"
exp2 = "figs/labels2/M134_20141204_v009/"
csv1 = "figs/labels/M134_20141204_v009/processed_grab.csv"
csv2 = "figs/labels2/M134_20141204_v009/processed_grab.csv"
dists, mismatch = post_processing.compare_csv_files(csv1, csv2)
label_dicts = post_processing.compare_experiment_consistency(exp1, exp2)

label_dicts = post_processing.compare_prediction_consistency(run1, run2)
for i in range(len(label_dicts)):
    print "label: %s" % label_dicts[i]["label"]
    print "\ttotal: %d" % label_dicts[i]["total"]
    print "\tmean: %f" % numpy.mean(label_dicts[i]["dists"])
    print "\tstd: %f" % numpy.std(label_dicts[i]["dists"])
    print "\tmismatched: %f" % label_dicts[i]["mismatched"]

# create an accuracy score for the run(need a better name for this).
# Let's do number tp/(tp + fp) + tp/(tp + fn), ie the f-score.
# in the output dictionary, fn are the false negatives, fp are the false
# positives and dists are the true positives.
print ""
print ""
for i in range(len(dict2)):
    tp = float(len(dict2[i]['dists']))
    fp = float(dict2[i]['fp'])
    fn = float(dict2[i]['fn'])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall)/(precision + recall)
    print "label: %s" % dict1[i]['label']
    print "\tprecision: %f" % precision
    print "\trecall: %f" % recall
    print "\tfscore: %f" % f1_score

# import pdb; pdb.set_trace()
