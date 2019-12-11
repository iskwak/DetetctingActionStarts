import os
import numpy
import sys
import helpers.post_processing as post_processing


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
        all_predict = numpy.zeros((5000, len(label_names)), dtype='float32')
        all_label = numpy.zeros((5000, len(label_names)), dtype='float32')
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
            all_predict, all_label, label_names, frame_thresh=30)
        # import pdb; pdb.set_trace()

        for label_i in range(len(label_names)):
            # if exps[i] == 'M134_20150325_v002':
            #     import pdb; pdb.set_trace()
            label_name = label_names[label_i]
            match_keys = all_matches[label_i].keys()
            current_match = all_matches[label_i]
            for key in match_keys:
                match_dict[label_name][key] += current_match[key]


def greedy_match(gt_idx, predict_sup, predict_idx, frame_thresh):
    # loop ever gt_idx, and check to see the nearest predict_idx
    match_dict = {
        "tps": [],
        "fps": [],
        "num_fn": 0
    }

    num_predict = len(predict_idx)
    # fps = list(range(num_predict))

    # create a list of sorted predictions by prediction score.
    predict_scores = []
    for j in range(len(predict_idx)):
        predict_scores.append(predict_sup[predict_idx[j]])
    # next sort the values
    sort_idx = numpy.argsort(predict_scores)[::-1]
    predict_idx = numpy.asarray(predict_idx)[sort_idx]
    fps = predict_idx.tolist()

    tps = []
    num_fn = 0
    for i in range(len(gt_idx)):
        # find the closest match.
        dists = []
        for j in range(len(predict_idx)):
            dists.append(numpy.abs(gt_idx[i] - predict_idx[j]))
        
        # loop over the distances and make tp's.
        is_matched = False
        for j in range(len(dists)):
            if dists[j] < frame_thresh:
                is_matched = True
                # this is a tp
                tps.append((gt_idx[i], predict_idx[j]))
                # remove this index from the fps.
                # check for multi match...
                if predict_idx[j] in fps:
                    fps.remove(predict_idx[j])
                # because this is a sorted list and analysis is not based
                # on what is closest but what is most confident, break
                # as soon as a match is found.
                break
        if is_matched is False:
            num_fn = num_fn + 1

    match_dict["tps"] = tps
    match_dict["fps"] = fps
    match_dict["num_fn"] = num_fn
    return match_dict


def analyze_outs(predict, labels, label_names, frame_thresh=30):
    # next apply non max suppression
    # labels = labels[0]
    num_labels = labels.shape[1]

    all_matches = []
    for i in range(num_labels):
        ground_truth = labels[:, i]
        gt_sup, gt_idx = post_processing.nonmax_suppress(
            ground_truth, 0.5)
        predict_sup, predict_idx = post_processing.nonmax_suppress(
            predict[:, i], 0.5)

        match_dict = greedy_match(gt_idx, predict_sup, predict_idx, frame_thresh)

        # match_dict, dist_mat = hungarian_matching.apply_hungarian(
        #     gt_idx, predict_idx
        # )
        match_dict["tps"] = len(match_dict["tps"])
        match_dict["fps"] = len(match_dict["fps"])
        # # write processed file
        # output_name = os.path.join(
        #     out_dir, exp_names[0], 'processed_%s.csv' % label_names[i]
        # )
        # create_proc_file(output_name, gt_sup, predict_sup, match_dict)
        all_matches.append(match_dict)
    return all_matches


def main():
    base_folder = "/nrs/branson/kwaki/outputs/thumos14/20190521-perframe_stop_0.5-perframe_0.5-loss_wasserstein-learning_rate_0.0001-decay_step_1-decay_0.9-anneal_type_none"
    predict_folder = os.path.join(base_folder, "predictions", "test")

    labels_fname = os.path.join(
        "/groups/branson/bransonlab/kwaki/data/thumos14/meta/labels.txt")
    
    label_names = []
    with open(labels_fname, "r") as fid:
        name = fid.readline().strip()
        while name:
            label_names.append(name)
            name = fid.readline().strip()
    
    match_dict = {}
    for key in label_names:
        match_dict[key] = {
            "tps": 0, "fps": 0, "num_fn": 0
        }

    
    process_exp(predict_folder, label_names, match_dict)


if __name__ == "__main__":
    main()
