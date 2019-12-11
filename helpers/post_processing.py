"""Some helper functions for post processing."""
from __future__ import print_function, division
import numpy
import csv
import os
import shutil
import scipy.optimize
# import matplotlib.pyplot as plt
import gflags
# import helpers.sequences_helper as sequences_helper
import helpers.hungarian_matching as hungarian_matching

gflags.DEFINE_float(
    "val_threshold", 0.70, "Threshold for classifier outputs.")
gflags.DEFINE_float(
    "frame_threshold", 10, "Distance to the positive example.")

# labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
# # labels = ["lift", "hand", "grab"]  # , "suppinate", "mouth", "chew"]


def nonmax_suppress(data, val_threshold):
    """Apply non maximal suppression."""
    # This search will go through each element once and check to see if it is
    # either the start of a local max bout, or if it is a local max.
    suppressed = numpy.zeros(data.shape)
    num_vals = len(data)
    in_bout = False
    bout_start = -1
    max_vals = []
    for i in range(1, num_vals):
        # first, check to see if in a "bout"
        if in_bout:
            # is this a bout that wasn't a max value.
            if data[i] > data[i - 1]:
                # start a new bout
                bout_start = i
            elif data[i] < data[i - 1]:
                # bout ends, and this was a local max bout.
                bout_end = i - 1
                # get the mid point of the bout
                mid_idx = int((bout_end - bout_start) / 2 + bout_start)
                suppressed[mid_idx] = data[mid_idx]
                max_vals.append(mid_idx)
                # end the bout
                in_bout = False
            # else the bout continues
        elif data[i] > data[i - 1] and data[i] > val_threshold:
            # we weren't in a bout, but one just started.
            bout_start = i
            in_bout = True
    return suppressed, max_vals


def load_predict_csv(filename):
    """Load a predict_*.csv file."""
    with open(filename, "r") as file:
        raw_data = list(csv.reader(file))

    # first convert the raw data into an nd-array... the list of
    # data per line is a bit rough.
    num_lines = len(raw_data) - 1
    num_cols = len(raw_data[0])

    fields = raw_data[0]
    # hack! if the last col is "image", pretend it doesn't exist.
    # if fields[-1] == "image":
    #     num_cols = num_cols - 1
    #     fields = fields[:-1]
    # bigger hack... remove image and nearest
    fields = [
        field for field in fields if field != 'nearest' and field != 'image'
    ]
    num_cols = len(fields)

    array_data = numpy.zeros((num_lines, num_cols))
    for i in range(1, len(raw_data)):
        # skip first row, its just the header
        for j in range(num_cols):
            if fields[j] != "image":
                array_data[i - 1, j] = raw_data[i][j]
    # convert to a dictionary.
    data = {}

    for i in range(len(fields)):
        data[fields[i]] = array_data[:, i]

    return data


def convert_csv_numpy(csv_data):
    """Convert data from the predict/process csv to a numpy data array."""
    csv_data = numpy.asarray(csv_data[1:])
    # convert the csv to a numpy array
    # data = numpy.zeros((csv_data.shape[0], csv_data.shape[1] - 1))
    data = numpy.zeros((csv_data.shape[0], 3))
    for i in range(csv_data.shape[0]):
        data[i] = [float(csv_data[i][j])
                   for j in range(csv_data[i].shape[0]) if j < 3]
    return data


def convert_csv_numpy2(csv_dict):
    """Convert data from the predict/process csv to a numpy data array."""
    # csv_data = numpy.asarray(csv_data[1:])
    csv_data = []
    for key in csv_dict.keys():
        if key != "frames":
            csv_data.append(csv_dict[key])
    # csv_data = numpy.concatenate(csv_data, axis=1)
    csv_data = numpy.vstack(csv_data).T

    # convert the csv to a numpy array
    # data = numpy.zeros((csv_data.shape[0], csv_data.shape[1] - 1))
    data = numpy.zeros((csv_data.shape[0], 3))
    for i in range(csv_data.shape[0]):
        data[i] = [float(csv_data[i][j])
                   for j in range(csv_data[i].shape[0]) if j < 3]
    return data, csv_data


def process_prediction_csv(filename, out_filename,
                           frame_thresh=5, val_threshold=0.75):
    """Given a predict_*.csv file, post process it."""
    # run non max on the labels and data
    all_csv = load_predict_csv(filename)
    # loop over the rows (2nd row and on) and convert to numeric
    # remove header
    data, all_csv = convert_csv_numpy2(all_csv)
    all_csv = data
    # all_csv = numpy.asarray(all_csv[1:])

    processed = data
    # predicted
    suppressed, max_vals = nonmax_suppress(data[:, 1], val_threshold)
    processed[:, 1] = suppressed
    # import pdb; pdb.set_trace()

    # ground truth
    suppressed, _ = nonmax_suppress(data[:, 2], val_threshold)
    processed[:, 2] = suppressed

    labelled = numpy.argwhere(processed[:, 2] == 1)
    labelled = labelled.flatten().tolist()
    # add a dummy variable for each of the nodes.
    # dist_mat = create_frame_dists(processed, max_vals, labelled)
    # num_found = len(max_vals)
    num_labelled = len(labelled)
    # rows, cols, dist_mat = apply_hungarian(dist_mat, frame_thresh)
    match_dict, dist_mat = hungarian_matching.apply_hungarian(
        max_vals, labelled
    )

    # missed classifications
    # false_neg = len(labelled) - len(
    #     [i for i in range(len(max_vals)) if cols[i] < len(labelled)])
    false_neg = match_dict["num_fn"]

    # extra predictions
    # for each idx in labelled, find its location in the cols array. The
    # location in the cols array represents the row id, which is related
    # to the max_vals ids.
    # false_pos = len(max_vals) - len(
    #     [i for i in range(len(labelled))
    #      if numpy.where(i == cols)[0][0] < len(max_vals)])
    false_pos = len(match_dict["fps"])

    # create a graph with this data
    with open(out_filename, "w") as file:
        # write the header first
        file.write("frame,predicted,ground truth,image,nearest\n")
        for i in range(all_csv.shape[0]):
            file.write("%f,%f,%f,%s" %
                       (processed[i, 0], processed[i, 1],
                        processed[i, 2], "frames/%05d.jpg" % i))

            if i in max_vals:
                # this is dumb...
                idx = numpy.where(numpy.asarray(max_vals) == i)
                # col_idx = cols[idx][0]
                # row_idx = rows[idx][0]
                # import pdb; pdb.set_trace()
                for j in range(len(match_dict["tps"])):
                    if i == match_dict["tps"][j][0]:
                        match = match_dict["tps"][j][1]
                        file.write(",%d" % match)
                    else:
                        file.write(",no match")

                # if col_idx < len(labelled) and\
                #         dist_mat[row_idx, col_idx] < frame_thresh:
                #     match = processed[labelled[col_idx], 0]
                #     file.write(",%d" % match)
                # else:
                #     # this peak does not have a matched peak
                #     file.write(",no match")
            # if i in max_vals:
            #     # if "M134_20141203_v030" in out_filename\
            #     #         and "hand" in out_filename:
            #     #     import pdb; pdb.set_trace()
            #     # is this a peak?
            #     # if it is a peak, does it have a match
            #     idx = numpy.where(numpy.asarray(max_vals) == i)
            #     if cols[idx][0] >= len(labelled):
            #         # this peak does not have a matched peak
            #         file.write(",no match")
            #     else:
            #         match = processed[labelled[cols[idx][0]], 0]
            #         file.write(",%d" % match)
            else:
                file.write(",N/A")

            file.write("\n")
            # mean_abs_offset += numpy.abs(dist)
            # num_count += 1

    # dists = dist_mat[rows[list(range(len(max_vals)))], cols[list(range(len(max_vals)))]]
    # dists = dists[dists < frame_thresh]
    dists = [
        numpy.abs(match[1] - match[0]) for match in match_dict["tps"]
    ]
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # return processed, data, all_csv, num_labelled, dists, missed, extra
    return num_labelled, dists, false_neg, false_pos


def create_frame_dists(data, frames1, frames2):
    """Helper function to create frame distances."""
    # total = labelled.shape[0]
    dist_mat = numpy.zeros((len(frames1), len(frames2)))
    for i in range(len(frames1)):
        for j in range(len(frames2)):
            dist_mat[i, j] = numpy.abs(
                data[frames1[i], 0] - data[frames2[j], 0])
    return dist_mat


def process_outputs(base_dir, out_dir, labels, frame_thresh=[10, 10, 10, 10, 10, 10],
                    val_threshold=0.70):
    """Given an output folder, loop over the experiments and process them."""
    exp_dirs = os.listdir(base_dir)
    # labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    label_dict = []
    for i in range(len(labels)):
        label_dict.append({
            "label": labels[i],
            "total": 0,
            "fn": 0,  # false negative
            "fp": 0,  # false positive
            "dists": []  # distances
        })

    for exp_dir in exp_dirs:
        # print exp_dir
        full_exp = os.path.join(base_dir, exp_dir)

        # copy templates
        shutil.copy('templates/processed_main.js', full_exp)
        shutil.copy('templates/processed_lift.html', full_exp)
        shutil.copy('templates/processed_hand.html', full_exp)
        shutil.copy('templates/processed_grab.html', full_exp)
        shutil.copy('templates/processed_suppinate.html', full_exp)
        shutil.copy('templates/processed_mouth.html', full_exp)
        shutil.copy('templates/processed_chew.html', full_exp)

        # get all the predict_*.csv files.
        csv_files = os.listdir(full_exp)
        csv_files = [filename for filename in csv_files
                     if "predict_" in filename and ".csv" in filename]
        # sort the files
        csv_files.sort()
        # import pdb; pdb.set_trace()
        for j in range(len(csv_files)):
            csv_file = csv_files[j]
            # out_filename = os.path.join(out_dir, csv_file)
            basename = os.path.splitext(csv_file)[0]
            out_filename = "processed_" + basename.split("_")[1] + ".csv"
            out_filename = os.path.join(full_exp, out_filename)
            full_cvsname = os.path.join(full_exp, csv_file)
            num_labelled, dists, false_neg, false_pos = \
                process_prediction_csv(
                    full_cvsname, out_filename, frame_thresh=frame_thresh[j],
                    val_threshold=val_threshold)
            idx = [i for i in range(len(labels)) if labels[i] in csv_file]
            idx = idx[0]
            label_dict[idx]['total'] += num_labelled
            label_dict[idx]['fn'] += false_neg
            label_dict[idx]['fp'] += false_pos
            # label_dict[idx]['dists'] += dists.tolist()
            label_dict[idx]["dists"] += dists
    return label_dict


# def compare_prediction_consistency(run1, run2):
#     """Compare the consistency between two runs."""
#     run1_exp = os.listdir(run1)
#     run1_exp.sort()
#     run2_exp = os.listdir(run2)
#     run2_exp.sort()
#     # update_keys = ["total", "mismatched", "dists"]

#     assert run1_exp == run2_exp
#     label_dicts = create_label_dict()
#     for i in range(len(run1_exp)):
#         exp1 = os.path.join(run1, run1_exp[i])
#         exp2 = os.path.join(run2, run2_exp[i])
#         updated = compare_experiment_consistency(exp1, exp2)
#         for j in range(len(label_dicts)):
#             label_dicts[j]["total"] += updated[j]["total"]
#             label_dicts[j]["mismatched"] += updated[j]["mismatched"]
#             label_dicts[j]["dists"] += updated[j]["dists"]

#     return label_dicts


# def create_label_dict():
#     """Initialize the label dictionary."""
#     label_dicts = []
#     for i in range(len(labels)):
#         label_dicts.append({
#             "label": labels[i],
#             "total": 0,
#             "mismatched": 0,
#             "dists": []  # distances
#         })
#     return label_dicts


# def compare_experiment_consistency(exp1, exp2):
#     """Compare the consistency between two experiments."""
#     # labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
#     label_dicts = create_label_dict()
#     # For each pair of labels, compare the consistency.
#     for i in range(len(labels)):
#         # create the csv filename for each experiment
#         csv1 = os.path.join(exp1, "processed_%s.csv" % labels[i])
#         csv2 = os.path.join(exp2, "processed_%s.csv" % labels[i])

#         dists, num_mismatch = compare_csv_files(csv1, csv2)
#         label_dicts[i]["mismatched"] += num_mismatch
#         label_dicts[i]["dists"] += dists.tolist()

#     return label_dicts


def apply_hungarian(dist_mat, thresh):
    """Apply the hungarian algorithm on the distance matrix."""
    num_rows = dist_mat.shape[0]
    num_cols = dist_mat.shape[1]

    dist_mat = numpy.concatenate(
        [dist_mat,
         numpy.zeros((num_rows, num_rows)) + thresh], axis=1)
    dist_mat = numpy.concatenate(
        [dist_mat,
         numpy.zeros((num_cols,
                      num_rows + num_cols)) + thresh],
        axis=0)
    rows, cols = scipy.optimize.linear_sum_assignment(dist_mat)
    return rows, cols, dist_mat


# def compare_csv_files(csv1, csv2):
#     """Compare csv files of two labelled sets."""
#     # Assume these are post processed files (ie, already non max suppressed.)
#     val_threshold = 0.1
#     all_csv = load_predict_csv(csv1)
#     data1 = convert_csv_numpy(all_csv)

#     all_csv = load_predict_csv(csv2)
#     data2 = convert_csv_numpy(all_csv)

#     # sanity check
#     assert numpy.argwhere((data1[:, 0] == data2[:, 0]) is False).size == 0

#     # find the locations of the predictions
#     sup1, maxs1 = nonmax_suppress(data1[:, 1], val_threshold)
#     sup2, maxs2 = nonmax_suppress(data2[:, 1], val_threshold)

#     # create the frame distance matrix
#     dist_mat = create_frame_dists(data1, maxs1, maxs2)
#     thresh = 20
#     rows1, cols1, augmented1 = apply_hungarian(dist_mat, thresh)
#     rows2, cols2, augmented2 = apply_hungarian(dist_mat.T, thresh)

#     dists1 = augmented1[rows1[list(range(len(maxs1)))], cols1[list(range(len(maxs1)))]]
#     dists2 = augmented2[rows2[list(range(len(maxs2)))], cols2[list(range(len(maxs2)))]]

#     num_missed1 = len(maxs2) - len(
#         [i for i in range(len(maxs1)) if rows1[i] < len(maxs2)])

#     # extra predictions
#     num_missed2 = len(maxs1) - len(
#         [i for i in range(len(maxs2)) if cols1[i] < len(maxs1)])

#     num_mismatch = num_missed1 + num_missed2

#     if len(maxs1) != len(maxs2):
#         dists1 == dists2
#     #     import pdb; pdb.set_trace()

#     dists1 = dists1[dists1 < thresh]
#     return dists1, num_mismatch


# def create_precision_recall_data(base_dir):
#     """Create a precision recall curve for processed outputs."""
#     frame_threshs = list(range(1, 20))
#     label_dicts = []
#     for thresh in frame_threshs:
#         label_dicts.append(
#             process_outputs(base_dir, None, frame_thresh=thresh))

#     mean_f = 0
#     for i in range(len(label_dicts)):
#         tp = float(len(label_dicts[i]['dists']))
#         fp = float(label_dicts[i]['fp'])
#         fn = float(label_dicts[i]['fn'])
#         precision = tp / (tp + fp + 0.0001)
#         recall = tp / (tp + fn + 0.0001)
#         f1_score = 2 * (precision * recall) / (precision + recall + 0.0001)
#         print("label: %s" % label_dicts[i]['label'])
#         print("\tprecision: %f" % precision)
#         print("\trecall: %f" % recall)
#         print("\tfscore: %f" % f1_score)
#         mean_f += f1_score
#     print("mean score: %f" % (mean_f / len(label_dicts)))

#     return label_dicts, frame_threshs


# def create_precision_recall_curve(out_dir, dicts, frame_threshs):
#     """Create a csv file for the precision recall curve."""
#     # for each label, create the csv data.
#     # for each threshold
#     # for each label
#     # create csv file for each label
#     values = []
#     for j in range(len(labels)):
#         values.append({
#             "label": labels[j],
#             "tp": [],
#             "fp": [],
#             "fn": [],
#             "precision": [],
#             "recall": []
#         })

#     for i in range(len(frame_threshs)):
#         for j in range(len(dicts[i])):
#             # if i == 12 or i == 3:
#             #     import pdb; pdb.set_trace()
#             # this is the loop over the labels
#             tp = float(len(dicts[i][j]["dists"]))
#             fp = float(dicts[i][j]["fp"])
#             fn = float(dicts[i][j]["fn"])
#             values[j]["tp"].append(tp)
#             values[j]["fp"].append(fp)
#             values[j]["fn"].append(fn)
#             values[j]["precision"].append(tp / (tp + fp))
#             values[j]["recall"].append(tp / (tp + fn))
#     # values collected, create the curves
#     for i in range(len(values)):
#         plt.plot(values[i]["recall"], values[i]["precision"], "bo-")
#         plt.show()


def process_outputs2(base_dir, out_dir, label_names,
                     frame_thresh=[10, 10, 10, 10, 10, 10],
                     val_threshold=0.70):
    """Given an output folder, loop over the experiments and process them."""
    exp_dirs = os.listdir(base_dir)
    # labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    label_dict = []
    for i in range(len(label_names)):
        label_dict.append({
            "label": label_names[i],
            "total": 0,
            "fn": 0,  # false negative
            "fp": 0,  # false positive
            "dists": []  # distances
        })

    for exp_dir in exp_dirs:
        # print exp_dir
        full_exp = os.path.join(base_dir, exp_dir)

        # get all the predict_*.csv files.
        csv_files = os.listdir(full_exp)
        csv_files = [filename for filename in csv_files
                     if "predict_" in filename and ".csv" in filename]
        # sort the files
        csv_files.sort()
        # import pdb; pdb.set_trace()
        for j in range(len(csv_files)):
            csv_file = csv_files[j]
            # out_filename = os.path.join(out_dir, csv_file)
            basename = os.path.splitext(csv_file)[0]
            out_filename = "processed_" + basename.split("_")[1] + ".csv"
            out_filename = os.path.join(full_exp, out_filename)
            full_cvsname = os.path.join(full_exp, csv_file)
            num_labelled, dists, false_neg, false_pos = \
                process_prediction_csv(
                    full_cvsname, out_filename, frame_thresh=frame_thresh[j],
                    val_threshold=val_threshold)

            idx = [i for i in range(len(label_names)) if label_names[i] in csv_file]
            idx = idx[0]
            label_dict[idx]['total'] += num_labelled
            label_dict[idx]['fn'] += false_neg
            label_dict[idx]['fp'] += false_pos
            # label_dict[idx]['dists'] += dists.tolist()
            import pdb; pdb.set_trace()
            label_dict[idx]['dists'] = dists
    return label_dict
