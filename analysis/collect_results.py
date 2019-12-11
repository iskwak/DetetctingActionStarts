"""Helper script to combine results."""

import os
import numpy
import sys
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching


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


def analyze_outs(predict, labels, label_names, frame_thresh=0.7):
    # next apply non max suppression
    # labels = labels[0]
    num_labels = labels.shape[1]

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


def process_exp(exp_dir, label_names, match_dict):
    exps = os.listdir(exp_dir)
    exps.sort()

    for i in range(len(exps)):
        # print(exps[i])
        # exps[i] = 'M134_20150325_v002'
        all_files = os.listdir(os.path.join(exp_dir, exps[i]))
        all_predict = numpy.zeros((1500, 6), dtype='float32')
        all_label = numpy.zeros((1500, 6), dtype='float32')
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
            all_predict[:num_rows, label_i] = predict
            all_label[:num_rows, label_i] = labels
        all_matches = analyze_outs(
            all_predict, all_label, label_names, frame_thresh=0.70)
        # import pdb; pdb.set_trace()

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
        'loss': 'hung',
        # 'loss': 'rgb'
        # 'loss': 'front'
        # 'val': 'perframe_0.5'
    }
    # feature = 'finetune2/'
    # feature = 'hoghof'
    # feature = 'all'
    # feature = 'rgb'
    feature = 'finetune2/baselr2'
    # feature = 'finetune2/smalllr2'
    # feature = 'canned_i3d'
    base_dir = '/nrs/branson/kwaki/outputs'
    # base_dir = '/nrs/branson/kwaki/outputs/i3d_ff2'
    # base_dir = '/nrs/branson/kwaki/outputs/finetune'
    # base_dir = '/nrs/branson/kwaki/outputs/tests'
    # base_dir = '/nrs/branson/kwaki/outputs/mouse_odas_reweight/'
    # base_dir = "/nrs/branson/kwaki/outputs/mouse_odas_reweight_unfroze/"
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


    # for label_name in label_names:
    #     print(label_name)
    #     print("\t%f, %f" % (numpy.mean(fscores[label_name]), numpy.std(fscores[label_name])))
    #     print("\t\t%f, %f, %f, %f, %f" %
    #           (numpy.mean(label_scores[label_name]["tp"]),
    #            numpy.mean(label_scores[label_name]["fp"]),
    #            numpy.mean(label_scores[label_name]["fn"]),
    #            numpy.mean(label_scores[label_name]["pre"]),
    #            numpy.mean(label_scores[label_name]["rec"])))
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
