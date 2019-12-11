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
        keys = params.keys()
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

    all_dists = {
        'lift': [],
        'hand': [],
        'grab': [],
        'supinate': [],
        'mouth': [],
        'chew': []
    }
    dist_keys = label_names
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
        # predict_idx = match_dict["tps"]
        predict_idx = match_dict["fps"]
        # setup "greedy matching" to make a list of distances for each prediction.
        dist_mat = numpy.zeros((len(gt_idx), len(predict_idx)))
        abs_dist_mat = numpy.zeros((len(gt_idx), len(predict_idx)))
        for j in range(len(gt_idx)):
            for k in range(len(predict_idx)):
                # dist_mat[j, k] = predict_idx[k][1] - gt_idx[j]
                # abs_dist_mat[j, k] = abs(predict_idx[k][1] - gt_idx[j])

                dist_mat[j, k] = predict_idx[k] - gt_idx[j]
                abs_dist_mat[j, k] = abs(predict_idx[k] - gt_idx[j])
        # get the min idx for each column (each prediction).
        min_dists = []
        if len(gt_idx) > 0 and len(predict_idx) > 0:
            min_idx = numpy.argmin(abs_dist_mat, axis=0)

            for j in range(len(predict_idx)):
                min_dists.append(dist_mat[min_idx[j], j])
            all_dists[dist_keys[i]] += min_dists
        # if len(gt_idx) > 1 and len(predict_idx) > 1:
        #     import pdb; pdb.set_trace()
        # match_dict, dist_mat = hungarian_matching.apply_hungarian(
        #     gt_idx, predict_idx
        # )
        # if len(match_dict["tps"]) > 0:
        #     import pdb; pdb.set_trace()
        # match_dict["tps"] = len(match_dict["tps"])
        # match_dict["fps"] = len(match_dict["fps"])
        # # write processed file
        # output_name = os.path.join(
        #     out_dir, exp_names[0], 'processed_%s.csv' % label_names[i]
        # )
        # create_proc_file(output_name, gt_sup, predict_sup, match_dict)
        # all_matches.append(dist_mat)
    return all_dists


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


def process_exp(exp_dir, label_names, dist_dict):
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
            all_predict[:, label_i] = predict
            all_label[:, label_i] = labels

        all_matches = analyze_outs(
            all_predict, all_label, label_names, frame_thresh=0.7)
        # import pdb; pdb.set_trace()

        for label_i in range(len(label_names)):
            # if exps[i] == 'M134_20150325_v002':
            #     import pdb; pdb.set_trace()
            label_name = label_names[label_i]
            dist_dict[label_name] += all_matches[label_name]


def main(argv):
    """main"""
    mice = ['M134', 'M147', 'M173', 'M174']
    params = {
        'loss': 'weighted',
        # 'val': 'perframe_0.5'
    }
    # feature = 'finetune2/'
    feature = 'hoghof'
    # feature = 'finetune2/baselr2'
    # feature = 'finetune2/smalllr2'
    # feature = 'canned_i3d'
    base_dir = '/nrs/branson/kwaki/outputs'
    # base_dir = '/nrs/branson/kwaki/outputs/tests'
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

    # label_scores = {}
    # for key in label_names:
    #     label_scores[key] = {
    #         'tp': [],
    #         'fp': [],
    #         'fn': [],
    #         'pre': [],
    #         'rec': [],
    #         'fscore': []
    #     }
    all_dists = {
        'lift': [],
        'hand': [],
        'grab': [],
        'supinate': [],
        'mouth': [],
        'chew': []
    }

    for mouse in mice:
        print(mouse)
        mouse_dir = os.path.join(base_dir, mouse, feature)
        exp_dir = get_main_exp_folder(mouse_dir, params)

        print(os.path.join(mouse_dir, exp_dir))
        test_dir = os.path.join(mouse_dir, exp_dir, "predictions", "test")
        process_exp(test_dir, label_names, all_dists)

        # compute the fscore
    numpy.save("/nrs/branson/kwaki/outputs/analysis/histogram/fps/%s.npy" % params["loss"],
               all_dists)
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
