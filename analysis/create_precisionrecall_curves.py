"""Helper script to create precision recall curves for the behaviors."""

import os
import numpy
import sys
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching
import helpers.arg_parsing as arg_parsing
import gflags

# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("loss", None, "Loss to analyze.")
gflags.DEFINE_string("base_dir", "/nrs/branson/kwaki/outputs", "Base directory to process.")
gflags.DEFINE_string("mode", "test", "Train or testing data.")
gflags.DEFINE_string("features", None, "Feature type.")


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    return opts


def get_main_exp_folder(base_dir, opts):
    param = opts["flags"].loss

    exps = os.listdir(base_dir)
    for exp in exps:
        # import pdb; pdb.set_trace()
        # all_keys = numpy.zeros((len(params), 1))
        # keys = params.keys()
        # for i in range(len(keys)):
        #     key = keys[i]
        #     if params[key] in exp:
        #         all_keys[i] = 1
        # if all_keys.sum() == len(params):
        #     return exp
        if param in exp:
            return exp


def analyze_outs(predict, labels, label_names, frame_thresh=0.7, dist_threshold=10):
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
            gt_idx, predict_idx, dist_threshold=dist_threshold
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


def process_exp(exp_dir, label_names, match_dict, frame_dist):
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
            all_predict, all_label, label_names, frame_thresh=0.7,
            dist_threshold=frame_dist
        )

        for label_i in range(len(label_names)):
            # if exps[i] == 'M134_20150325_v002':
            #     import pdb; pdb.set_trace()
            label_name = label_names[label_i]
            match_keys = all_matches[label_i].keys()
            current_match = all_matches[label_i]
            for key in match_keys:
                match_dict[label_name][key] += current_match[key]



def process_mice(opts, mice, label_names, frame_dist):
    """Process a mice at a certain threshold."""
    label_scores = {}
    # want to log each behavior's stats seperately.
    for j in range(len(label_names)):
        label_scores[label_names[j]] = {
            'tp': [],
            'fp': [],
            'fn': [],
            'precision': [],
            'recall': [],
            'fscore': []
        }

    for mouse in mice:
        # print(mouse)
        mouse_dir = os.path.join(opts["flags"].base_dir, mouse, opts["flags"].features)
        exp_dir = get_main_exp_folder(mouse_dir, opts)

        match_dict = {}
        for key in label_names:
            match_dict[key] = {
                "tps": 0, "fps": 0, "num_fn": 0
            }
        print(os.path.join(mouse_dir, exp_dir))
        test_dir = os.path.join(mouse_dir, exp_dir, "predictions", opts["flags"].mode)
        process_exp(test_dir, label_names, match_dict, frame_dist)

        # compute the fscore
        for label_name in label_names:
            # print(label_name)
            tp = match_dict[label_name]["tps"] * 1.0
            fp = match_dict[label_name]["fps"] * 1.0
            fn = match_dict[label_name]["num_fn"] * 1.0

            precision = (tp / (tp + fp))
            recall = (tp / (tp + fn))
            fscore = 2 * (precision * recall) / (precision + recall)

            label_scores[label_name]["tp"].append(tp)
            label_scores[label_name]["fp"].append(fp)
            label_scores[label_name]["fn"].append(fn)
            label_scores[label_name]["precision"].append(precision)
            label_scores[label_name]["recall"].append(recall)
            label_scores[label_name]["fscore"].append(fscore)

    # for each label create a "total" fscore
    for label_name in label_names:
        temp_dict= label_scores[label_name]
        temp_dict["total_tp"] = sum(temp_dict["tp"])
        temp_dict["total_fp"] = sum(temp_dict["fp"])
        temp_dict["total_fn"] = sum(temp_dict["fn"])

        temp_dict["total_precision"] =\
            temp_dict["total_tp"] / (temp_dict["total_tp"] + temp_dict["total_fp"])
        temp_dict["total_recall"] =\
            temp_dict["total_tp"] / (temp_dict["total_tp"] + temp_dict["total_fn"])
        temp_dict["total_fscore"] =\
            2 * (temp_dict["total_precision"] * temp_dict["total_recall"]) / \
            (temp_dict["total_precision"] + temp_dict["total_recall"])
        
    return label_scores


def main(argv):
    """main"""
    # params = {
    #     'loss': 'weighted'
    # }
    # feature = 'canned_i3d'
    # base_dir = '/nrs/branson/kwaki/outputs'
    opts = _setup_opts(argv)

    mice = ['M134', 'M147', 'M173', 'M174']
    label_names = [
        'lift', 'hand', 'grab', 'supinate', 'mouth', 'chew'
    ]
    params = {
        'loss': opts["flags"].loss
    }

    fscores = {}
    for key in label_names:
        fscores[key] = []

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

    # loop over the frame distances.
    frame_dists = [
        i for i in range(5, 55, 5)
    ]
    # add a frame dist of 1 to the list.
    # frame_dists.insert(0, 1)

    # for each threshold, create a struct containing the score information.
    lift_fid = open("/nrs/branson/kwaki/outputs/analysis/lift.cvs", "w")
    hand_fid = open("/nrs/branson/kwaki/outputs/analysis/hand.cvs", "w")
    grab_fid = open("/nrs/branson/kwaki/outputs/analysis/grab.cvs", "w")
    sup_fid = open("/nrs/branson/kwaki/outputs/analysis/sup.cvs", "w")
    mouth_fid = open("/nrs/branson/kwaki/outputs/analysis/mouth.cvs", "w")
    chew_fid = open("/nrs/branson/kwaki/outputs/analysis/chew.cvs", "w")
    fids = [
        lift_fid, hand_fid, grab_fid, sup_fid, mouth_fid, chew_fid
    ]
    with open("/nrs/branson/kwaki/outputs/analysis/temp.csv", "w") as fid:
        fid.write("threshold,lift,hand,grab,supinate,mouth,chew\n")
        score_dicts = []
        for i in range(len(frame_dists)):
            label_dict = process_mice(opts, mice, label_names, frame_dists[i])
            score_dicts.append(label_dict)

            fid.write("%f" % frame_dists[i])
            for label_name in label_names:
                fscore = label_dict[label_name]["total_fscore"]
                fid.write(",%f" % fscore)
            fid.write("\n")

            for j in range(len(fids)):
                fids[j].write(
                    "%f,%f\n" %
                    (label_dict[label_names[j]]["total_precision"],
                     label_dict[label_names[j]]["total_recall"]
                    ))

    # for each behavior, create a roc
    # for label_name in label_names:
    lift_fid.close()
    hand_fid.close()
    grab_fid.close()
    sup_fid.close()
    mouth_fid.close()
    chew_fid.close()
            
if __name__ == "__main__":
    main(sys.argv)

