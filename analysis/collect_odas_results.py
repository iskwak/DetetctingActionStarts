"""Helper script to combine results."""

import os
import numpy
import sys
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching
import helpers.sequences_helper as sequences_helper
import helpers.paths as paths


def get_main_exp_folder(base_dir, params):
    exps = os.listdir(base_dir)
    for exp in exps:
        all_keys = numpy.zeros((len(params), 1))
        keys = list(params.keys())
        for i in range(len(keys)):
            key = keys[i]
            if params[key] in exp:
                all_keys[i] = 1
        if all_keys.sum() == len(params):
            return exp


def analyze_outs(csv_dir, predict, labels, label_names, frame_thresh=0.1):
    # next apply non max suppression
    # labels = labels[0]
    num_labels = labels.shape[1]

    # create a mask for the network predictions.
    # apply an argmax to figure out which locations the network was most sure
    # that a behavior has occured
    predict_mask = numpy.zeros(predict.shape)
    predict_max_idx = numpy.argmax(predict, axis=1)
    for i in range(len(predict_max_idx)):
        # "i" should be the row.
        j = predict_max_idx[i]
        predict_mask[i, j] = 1

    # next convert the mask into predictions using the rules described by
    # startnet/odas paper.
    # c_t = argmax is an action.
    # c_t != c_{t-1}
    # as_t^{c_t} exceeds a threshold. This isn't available for odas.
    # set the previous behavior to background
    c_tminus1 = 6
    predict_starts = numpy.zeros(predict.shape)
    for i in range(predict_mask.shape[0]):
        c_t = predict_max_idx[i]
        if c_t != 6 and c_t != c_tminus1:
            predict_starts[i, c_t] = 1
        c_tminus1 = c_t

    # write predictions back to disk
    # copy the templates
    for label_name in label_names:
        csv_name = "odas_" + label_name + ".csv"
        base_out = csv_dir
        sequences_helper.create_html_file(
            base_out, csv_name, "movie_comb.avi", 30
        )
        # write the predictions.
    exp_name = os.path.basename(base_out)
    write_csvs(base_out, exp_name, label_names, labels, predict_starts)

    all_matches = []
    for i in range(num_labels):
        ground_truth = labels[:, i]
        gt_sup, gt_idx = post_processing.nonmax_suppress(
            ground_truth, frame_thresh)
        predict_sup, predict_idx = post_processing.nonmax_suppress(
            predict[:, i], frame_thresh)
        match_dict, dist_mat = hungarian_matching.apply_hungarian(
            gt_idx, predict_idx
        )
        match_dict["tps"] = len(match_dict["tps"])
        match_dict["fps"] = len(match_dict["fps"])
        # # write processed file
        # output_name = os.path.join(
        #     out_dir, exp_names[0], 'processed_%s.csv' % label_names[i]
        # )
        # create_proc_file(output_name, gt_sup, predict_sup, match_dict)
        all_matches.append(match_dict)
    return all_matches


def write_csvs(out_dir, exp_name, label_names, labels, predict):
    # frame, behavior, behavior ground truth, image
    # labels = labels.reshape((labels.shape[0], 1, labels.shape[1]))
    # predict = predict.reshape((predict.shape[0], 1, predict.shape[1]))
    frames = [list(range(labels.shape[0]))]
    temp = [
        label for label in label_names
    ]
    # for each prediction, update the csv file.
    current_exp_path = out_dir # os.path.join(out_dir, exp_name)
    paths.create_dir(out_dir)
    paths.create_dir(current_exp_path)

    for j in range(len(temp)):
        # filename = "%03d_predict_%s.csv" % (j, labels[j])
        filename = "odas_%s.csv" % temp[j]
        current_exp_file = os.path.join(current_exp_path,
                                        filename)
        with open(current_exp_file, "w") as outfile:
            sequences_helper.write_csv(
                outfile,
                temp[j],
                predict[:, j],
                labels[:, j],
                frames[0])


def read_csv(filename):
    predict = []
    labels = []
    with open(filename, "r") as fid:
        # ignore first line.
        fid.readline()
        for line in fid:
            data = line.split(",")
            predict.append(float(data[1]))
            labels.append(float(data[2]))
    return numpy.asarray(predict), numpy.asarray(labels)


def read_all_csv(filename):
    predict = []
    with open(filename, "r") as fid:
        fid.readline()
        for line in fid:
            data = line.split(",")
            for j in range(len(data)):
                data[j] = float(data[j])
            predict.append(numpy.asarray(data))
    return numpy.asarray(predict)


def process_exp(exp_dir, label_names, match_dict):
    exps = os.listdir(exp_dir)
    exps.sort()

    for i in range(len(exps)):
        # print(exps[i])
        # exps[i] = 'M134_20150325_v002'
        all_files = os.listdir(os.path.join(exp_dir, exps[i]))
        # all_predict = numpy.zeros((1500, 7), dtype='float32')
        # labels have 1 less dimension...
        all_label = numpy.zeros((1500, 6), dtype='float32')
        # get the labels from the
        for label_i in range(len(label_names)):
            label = label_names[label_i]
            idx = [
                j for j in range(len(all_files))
                if label in all_files[j] and "predict_" in all_files[j]
                and "html" not in all_files[j]
            ]
            # print(all_files[idx[0]])
            # load the file
            if len(idx) < 1:
                break
            csv_name = os.path.join(exp_dir, exps[i], all_files[idx[0]])
            predict, labels = read_csv(csv_name)
            num_rows = predict.shape[0]
            # all_predict[:num_rows, label_i] = predict
            all_label[:num_rows, label_i] = labels
        csv_name = os.path.join(exp_dir, exps[i], "all.csv")
        all_predict = read_all_csv(csv_name)
        csv_dir = os.path.join(exp_dir, exps[i])
        all_matches = analyze_outs(
            csv_dir, all_predict, all_label, label_names, frame_thresh=0.80)

        for label_i in range(len(label_names)):
            # if exps[i] == 'M134_20150325_v002':
            #     import pdb; pdb.set_trace()
            label_name = label_names[label_i]
            match_keys = all_matches[label_i].keys()
            current_match = all_matches[label_i]
            for key in match_keys:
                match_dict[label_name][key] += current_match[key]



def main(argv):
    """main"""
    mice = ['M134', 'M147', 'M173', 'M174']
    params = {
        'loss': 'rgb'
    }
    feature = 'all'
    # base_dir = '/nrs/branson/kwaki/outputs/mouse_odas_reweight/'
    base_dir = "/nrs/branson/kwaki/outputs/mouse_odas_reweight_unfroze/"
    label_names = [
        'lift', 'hand', 'grab', 'supinate', 'mouth', 'chew'
    ]

    fscores = {}
    # for mouse in mice:
    #     fscores[mouse] = {}
    for key in label_names:
        # fscores[mouse][key] = []
        fscores[key] = []
        # {'fscores': []}

    label_scores = {}
    for key in label_names:
        label_scores[key] = {
            'tp': [],
            'fp': [],
            'fn': [],
            'pre': [],
            'rec': [],
            'fscore': []
        }

    for mouse in mice:
        print(mouse)
        mouse_dir = os.path.join(base_dir, mouse, feature)
        exp_dir = get_main_exp_folder(mouse_dir, params)

        match_dict = {}
        for key in label_names:
            match_dict[key] = {
                "tps": 0, "fps": 0, "num_fn": 0
            }
        print(os.path.join(mouse_dir, exp_dir))
        test_dir = os.path.join(mouse_dir, exp_dir, "predictions", "test")
        process_exp(test_dir, label_names, match_dict)

        # compute the fscore
        for label_name in label_names:
            # print(label_name)
            tp = match_dict[label_name]["tps"] * 1.0
            fp = match_dict[label_name]["fps"] * 1.0
            fn = match_dict[label_name]["num_fn"] * 1.0
            precision = (tp / (tp + fp))
            recall = (tp / (tp + fn))
            if (precision + recall) == 0:
                fscore = 0
            else:
                fscore = 2 * (precision * recall) / (precision + recall)
            fscores[label_name].append(fscore)

            label_scores[label_name]["tp"].append(tp)
            label_scores[label_name]["fp"].append(fp)
            label_scores[label_name]["fn"].append(fn)
            label_scores[label_name]["pre"].append(precision)
            label_scores[label_name]["rec"].append(recall)
            label_scores[label_name]["fscore"].append(fscore)

        # print("\tPrecision: %f" % precision)
        # print("\tRecall: %f" % recall)
        # print("\tF score: %f" % (2 * (precision * recall) / (precision + recall)))
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for label_name in label_names:
        print(label_name)

        tp = numpy.sum(label_scores[label_name]["tp"])
        fp = numpy.sum(label_scores[label_name]["fp"])
        fn = numpy.sum(label_scores[label_name]["fn"])

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = (tp / (tp + fp))
        recall = (tp / (tp + fn))
        fscore = 2 * (precision * recall) / (precision + recall)
        print("\t%f" % fscore)
        print("\t\t%f, %f, %f, %f, %f" %
              (numpy.sum(label_scores[label_name]["tp"]),
               numpy.sum(label_scores[label_name]["fp"]),
               numpy.sum(label_scores[label_name]["fn"]),
               precision, recall))

    print("total")
    precision = (total_tp / (total_tp + total_fp))
    recall = (total_tp / (total_tp + total_fn))
    fscore = 2 * (precision * recall) / (precision + recall)
    print("\t%f" % fscore)
    print("\t\t%f, %f, %f, %f, %f" %
            (total_tp,
            total_fp,
            total_fn,
            precision, recall))


if __name__ == "__main__":
    main(sys.argv)
