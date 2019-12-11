"Helper to analyze outputs after processing."
import os
import sys
import shutil
import datetime
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching


def check_tps(frame_num, tps_list):
    # Helper function to find if a there is a match for the current frame.
    for sample in tps_list:
        if frame_num == sample[1]:
            return sample[0]
    return -1


def create_proc_file(output_name, gt, predict, match_dict):
    """create_post_proc_file

    Create post processed version of the file with matches.
    """
    header_str = "frame,predicted,ground truth,image,nearest\n"
    with open(output_name, "w") as fid:
        # write header
        fid.write(header_str)
        num_lines = len(gt)
        for i in range(num_lines):
            fid.write("%f,%f,%f,notused," % (i, predict[i], gt[i]))
            if i in match_dict["fps"]:
                # write false positive
                fid.write("no match")
            else:
                match_frame = check_tps(i, match_dict["tps"])
                if match_frame != -1:
                    fid.write("%d" % i)
                else:
                    fid.write("N/A")
            fid.write("\n")
    # import pdb; pdb.set_trace()


def proc_prediction_file(predict_file):
    """proc_prediction

    Process a single prediction file. Given a csv output of the form
    frame num, prediction, ground truth.
    """
    # Post processing flow:
    #  nonmax suppress
    #  Apply hungarian matching
    #  Compute stats.
    csv_data = post_processing.load_predict_csv(predict_file)
    # always assume there is a ground truth key name, and skip anything called
    # frames
    key_names = csv_data.keys()

    # get the values out of the dictionary
    ground_truth = []
    predict = []
    for key in key_names:
        if "ground truth" in key:
            ground_truth = csv_data[key]
        elif "frame" not in key:
            predict = csv_data[key]

    # next apply non max suppression
    gt_sup, gt_idx = post_processing.nonmax_suppress(ground_truth, 0.7)
    predict_sup, predict_idx = post_processing.nonmax_suppress(predict, 0.7)

    # hungarian matching
    match_dict, dist_mat = hungarian_matching.apply_hungarian(
        gt_idx, predict_idx
    )

    # create the post processed file
    output_name = predict_file.replace("predict_", "processed_")
    create_proc_file(output_name, gt_sup, predict_sup, match_dict)

    return match_dict


def proc_all_predictions(predict_dir):
    """proc_all_predictions

    Given a directory of predict, create post processed predictions if
    neccessary. Return stats on the experiment.
    """
    # get all the predict_*.csv
    all_files = os.listdir(predict_dir)
    predict_csvs = [
        csv for csv in all_files if "predict_" in csv and ".csv" in csv
    ]
    predict_csvs.sort()
    # match count updates
    # need one for all counts. and an array for each class.
    all_counts = {
        "tps": 0,
        "fps": 0,
        "fns": 0
    }
    # match_counts = {
    #
    # }
    for csv in predict_csvs:
        full_csv_name = os.path.join(predict_dir, csv)
        match_dict = proc_prediction_file(full_csv_name)
        # update all_counts
        all_counts["tps"] += len(match_dict["tps"])
        all_counts["fps"] += len(match_dict["fps"])
        all_counts["fns"] += len(match_dict["fns"])

    return all_counts


def setup_output_dir(data_dir, out_dir):
    """Setup the output space."""
    org_test_dir = data_dir
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_test_dir = os.path.join(out_dir, current_date)

    # setup the new test area. copy the folders in org_test_dir into
    # the new test dir.
    os.makedirs(new_test_dir)
    sub_folders = os.listdir(org_test_dir)

    for sub_folder in sub_folders:
        org_dir = os.path.join(org_test_dir, sub_folder)
        new_dir = os.path.join(new_test_dir, "data", sub_folder)
        shutil.copytree(org_dir, new_dir)

    plots_dir = os.path.join(new_test_dir, "plots")
    os.makedirs(plots_dir)

    return new_test_dir


def main(argv):
    """Main function.

    Given a folder of prediciton outputs, analyze the predictions and processed
    values.
    """
    test_dir = setup_output_dir(argv[1], argv[2])
    predictions_base_dir = os.path.join(test_dir, "data")
    predictions_dirs = os.listdir(predictions_base_dir)
    predictions_dirs.sort()

    all_counts = {
        "tps": 0,
        "fps": 0,
        "fns": 0
    }
    for prediction_dir in predictions_dirs:
        full_predict_name = os.path.join(predictions_base_dir, prediction_dir)
        counts = proc_all_predictions(full_predict_name)
        all_counts["tps"] += counts["tps"]
        all_counts["fps"] += counts["fps"]
        all_counts["fns"] += counts["fns"]

    return


if __name__ == "__main__":
    main(sys.argv)
