import numpy
import numpy as np
import os
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching
import sklearn.metrics


def get_label_names():
    labels_fname = os.path.join(
        "/groups/branson/bransonlab/kwaki/data/thumos14/meta/labels.txt")
    label_names = []
    with open(labels_fname, "r") as fid:
        name = fid.readline().strip()
        while name:
            label_names.append(name)
            name = fid.readline().strip()
    return label_names


def read_csv(filename):
    net_out = []
    gt = []
    with open(filename, "r") as fid:
        # get header
        line = fid.readline()
        line = fid.readline()
        # order:
        # frame num, predict, gt, image
        while line:
            data = line.split(',')
            # print(data)

            net_out.append(float(data[1]))
            gt.append(float(data[2]))

            line = fid.readline()

    # combine
    net_out = numpy.asarray(net_out)
    gt = numpy.asarray(gt)
    scores = numpy.stack([net_out, gt]).T

    return scores


def eval_scores(scores, score_thresh=0.7, dist_thresh=30.0):
    gt_sup, gt_idx = post_processing.nonmax_suppress(
        scores[:, 1], score_thresh)
    predict_sup, predict_idx = post_processing.nonmax_suppress(
        scores[:, 0], score_thresh)

    # go through each prediction and see if it is a false positive or true
    # positive.
    predicts = predict_sup[predict_idx]
    is_tp = numpy.zeros(predicts.shape)
    is_gt = numpy.zeros(predicts.shape)
    match_dict, dist_mat = hungarian_matching.apply_hungarian(
        gt_idx, predict_idx, val_threshold=score_thresh, dist_threshold=dist_thresh
    )

    # # loop over the predictions, and create a is_tp_fp indicator
    # for i in range(len(match_dict["tps"])):
    #     # second element is the prediction index
    #     is_tp[match_dict["tps"][i][1]] = 1
    # is_tp = is_tp[predict_idx]

    # for i in range(len(match_dict["fps"])):
    #     # second element is the prediction index
    #     is_tp[match_dict["tps"][i][1]] = 1
    #     # try:
    #     #     is_tp[match_dict["tps"][i][1] == numpy.asarray(predict_idx)] = 1
    #     # except:
    #     #     import pdb; pdb.set_trace()


    for j in range(len(predict_idx)):
        for i in range(len(gt_idx)):
            # check if in the right range?
            # dist mat is buffered by dummy nodes (of the number of gt's)
            if numpy.argmin(dist_mat[:, j + len(gt_idx)]) < dist_thresh:
                # is this prediction the lowest score for any gt?
                dists = dist_mat[i, :]
                if numpy.argmin(dists) == j + len(gt_idx):
                    is_tp[j] = dists[j + len(gt_idx)]
                    is_gt[j] = True
                    # no reason to check for other matches
                    break
            # else:
            #     is_tp[j] = -1
            #     import pdb; pdb.set_trace()
    # if len(gt_idx) > 2:
    #     import pdb; pdb.set_trace()
    return predicts, is_tp, is_gt, len(gt_idx)


# base_folder = "/nrs/branson/kwaki/outputs/thumos14/canned_64_bigger_win/20190725-perframe_stop_0.5-perframe_0.5-loss_weighted_mse-learning_rate_1e-05-label_key_end_labels-decay_step_1-decay_0.9-arch_concat-anneal_type_none/"
def main():
    base_folder = "/nrs/branson/kwaki/outputs/thumos14/canned_64_bigger_win/20190725-perframe_stop_0.5-perframe_0.5-loss_weighted_mse-learning_rate_1e-05-label_key_labels-decay_step_1-decay_0.9-arch_bidirconcat-anneal_type_none/"

    # for each prediction in a directory, label it as a false positive or
    # true positive. need two versions... one where things are cut out
    # if outside of a range, and not cut out.

    label_names = get_label_names()
    # label_names = [
    #     'lift', 'hand', 'grab', 'supinate', 'mouth', 'chew'
    # ]
    test_folder = os.path.join(base_folder, "predictions", "test")
    # test_folder = os.path.join(base_folder, "predictions", "train")
    exps = os.listdir(test_folder)
    exps.sort()

    # frame_dists = [
    #     i * 10 for i in range(1, 11)
    # ]
    frame_dists = [
        i * 30 for i in range(1, 11)
    ]

    exp_scores = []
    for i in range(len(exps)):
        # print(exps[i])
        exp_folder = os.path.join(test_folder, exps[i])
        files = os.listdir(exp_folder)
        # print(files)
        # use only the predict_*csv files
        files.sort()
        # for each file create a label matrix
        all_scores = []
        for label_name in label_names:
            predict_files = [
                predict_csv for predict_csv in files
                if "predict" in predict_csv and "csv" in predict_csv and
                    label_name in predict_csv
            ]

            all_scores.append(
                read_csv(os.path.join(exp_folder, predict_files[0]))
            )
        exp_scores.append(all_scores)
        # find the values that are predicted high, and then

    depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # depths = [1.0]
    for depth_i in range(len(depths)):
        print(depths[depth_i])
        # load all the data first
        mAP = []
        for label_i in range(len(label_names)):
            if label_i == 0:
                continue
            # print(label_names[label_i])
            total = 0
            class_map = []
            for frame_dist in frame_dists:
                all_predict = []
                all_is_tp = []
                all_is_gt = []
                all_pos = 0
                for i in range(len(exps)):
                    scores = exp_scores[i][label_i]
                    predictions, is_tp, is_gt, pos = eval_scores(scores, dist_thresh=frame_dist)
                    all_predict = all_predict + predictions.tolist()
                    all_is_tp = all_is_tp + is_tp.tolist()
                    all_is_gt = all_is_gt + is_gt.tolist()
                    all_pos = all_pos + pos

                # convert these to a list to compute the mean ap...
                sort_idx = numpy.argsort(1 - numpy.asarray(all_predict))
                sorted_predict = numpy.asarray(all_predict)[sort_idx]
                sorted_is_tp = numpy.asarray(all_is_tp)[sort_idx]
                sorted_is_gt = numpy.asarray(all_is_gt)[sort_idx]

                tp = numpy.cumsum(sorted_is_tp > 0) * 1.0
                fp = numpy.cumsum(~(sorted_is_tp > 0)) * 1.0
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                rec = tp / all_pos

                # last_score = sklearn.metrics.average_precision_score(
                #     sorted_is_gt, sorted_predict
                # )
                # if numpy.isnan(last_score):
                #     last_score = 0
                # print("\t%f" % compute_tpfp(sorted_is_tp))
                last_score = compute_tpfp(
                    sorted_is_tp, tp, fp, prec, rec, all_pos, depths[depth_i]
                )
                # last_score = voc_ap(rec, prec, depth=depths[depth_i])
                # import pdb; pdb.set_trace()
                # print("\t%f" % last_score)
                class_map.append(last_score)
            # import pdb; pdb.set_trace()
            mAP.append(sum(class_map) / len(frame_dists))
            # print("\t%f" % (sum(class_map) / len(frame_dists)))
            # mAP.append(last_score)
        print("\t%f" % (sum(mAP) / len(label_names)))


def compute_tpfp(tps, tp, fp, prec, rec, npos, depth):
    numer = 0
    # for i in range(len(tps)):
    #     if tps[i] == 0:
    #         numer = numer + 0
    #     else:
    #         # numer = numer + (sum(tps[:i+1]) / len(tps[:i+1]))
    #         numer = numer + (sum(tps[:i+1] > 0) * 1.0 / len(tps[:i+1]))
    num_pos = 0
    for i in range(len(tps)):
        if rec[i] > depth:
            break
        if tps[i] == 0:
            numer = numer + 0
        else:
            # numer = numer + (sum(tps[:i+1]) / len(tps[:i+1]))
            # numer = numer + (tp[i] / (tp[i] + fp[i]))
            numer = numer + prec[i]
            num_pos += 1

    # return numer / np.maximum(sum(tps > 0), np.finfo(np.float64).eps)
    # return numer / np.maximum(npos, np.finfo(np.float64).eps)
    # return numer / np.maximum( npos, np.finfo(np.float64).eps)
    return numer / np.maximum( num_pos, np.finfo(np.float64).eps)


def voc_ap(rec, prec, depth=1.0):
    end_depth = depth + 0.1
    # 11 point metric
    ap = 0.
    # count = len(np.arange(0., depth, 0.1))
    # for t in np.arange(0., depth, 0.1):
    count = len(np.arange(0., end_depth, 0.1))
    for t in np.arange(0., end_depth, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap
    # # not interpolated
    # # ap = 0
    # # for i in range(len(prec) - 1):
    # #     ap = ap + (prec[i] + prec[i +1]) / 2 * (rec[i + 1] - rec[i])
    # # correct AP calculation
    # # first append sentinel values at the end
    # mrec = np.concatenate(([0.], rec, [1.]))
    # mpre = np.concatenate(([0.], prec, [0.]))

    # idx = np.argmax(np.argwhere(mrec <= depth))
    # mrec = mrec[:idx+1]
    # mpre = mpre[:idx+1]

    # # compute the precision envelope
    # for i in range(mpre.size - 1, 0, -1):
    #     mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # # to calculate area under PR curve, look for points
    # # where X axis (recall) changes value
    # i = np.where(mrec[1:] != mrec[:-1])[0]

    # # and sum (\Delta recall) * prec
    # ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    # ap = mpre.mean()
    # return ap


if __name__ == "__main__":
    main()
