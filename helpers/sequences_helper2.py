"""Some helpers for sequence data."""
from __future__ import print_function, division
import numpy
import time
from . import paths
import os
import shutil
import platform
import helpers.template_helper as template_helper
import helpers.post_processing as post_processing


def log_outputs(opts, network, round_tic, train_cost, test_cost):
    """Create/update csv files."""
    t = network["lr_update"]["params"][0]
    if "total_iterations" in list(opts.keys()):
        total_iterations = opts["total_iterations"]
    else:
        total_iterations = opts["flags"].total_iterations
    print("(%d, %d)" % (t.get_value(), total_iterations))
    print("\titers time: %d seconds" % (time.time() - round_tic))
    print("\ttrain cost: %f" % train_cost)
    print("\ttest cost: %f" % test_cost)
    print("\tlearning rate: %f" % network["learning_rate"].eval())

    # write the training and validation numbers to disk
    if "out_dir" in list(opts.keys()):
        out_dir = opts["out_dir"]
    else:
        out_dir = opts["flags"].out_dir

    loss_filename = os.path.join(out_dir, "plots", "loss.csv")
    if os.path.isfile(loss_filename) is False:
        # file doesn"t exist, create the header
        with open(loss_filename, "w") as outfile:
            outfile.write("iteration,training loss,test loss\n")
    with open(loss_filename, "a") as outfile:
        outfile.write("%d,%f,%f\n" %
                      (t.get_value(), train_cost, test_cost))

    return


def write_csv(outfile, label_name, predict, ground_truth, frames):
    """Given a numpy array, create a csv file for it.

    Note, this adds a row idx output to the csv.
    NOTE: no longer generic, now helper function for write_predictions. And
    probably needs a new name...
    """
    # ... for now hard code label names
    # num_rows = predict.shape[0]
    num_rows = min(len(frames), predict.shape[0], ground_truth.shape[0])
    # write the header!
    outfile.write("frame,")
    outfile.write(label_name + ",")
    outfile.write(label_name + " ground truth,")
    outfile.write("image\n")

    for i in range(num_rows):
        # outfile.write("%f" % array[i][0])
        outfile.write("%f," % frames[i])
        outfile.write("%f," % predict[i])
        outfile.write("%f," % ground_truth[i])
        outfile.write("frames/%05d.jpg" % frames[i])
        # for j in range(num_cols):
        #     outfile.write(",%f" % predict[i][j])
        #     outfile.write(",%f" % ground_truth[i][j])
        outfile.write("\n")
    return


def write_predictions(out_dir, exp_names, predictions, ground_truth, frames):
    """Write predictions to disk."""
    labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    for i in range(len(predictions)):
        # for each prediction, update the csv file.
        current_exp_path = os.path.join(out_dir, exp_names[i][0])
        paths.create_dir(out_dir)
        paths.create_dir(current_exp_path)

        for j in range(len(labels)):
            # filename = "%03d_predict_%s.csv" % (j, labels[j])
            filename = "predict_%s.csv" % labels[j]
            current_exp_file = os.path.join(current_exp_path,
                                            filename)
            with open(current_exp_file, "w") as outfile:
                write_csv(outfile,
                          labels[j],
                          predictions[i][:, j],
                          ground_truth[i][:, j],
                          frames[i])


def write_predictions2(out_dir, exp_names, predictions, ground_truth,
                       hidden_list, frames, label_names=None):
    """Write predictions to disk."""
    if label_names is None:
        labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    else:
        labels = label_names
    for i in range(len(exp_names)):
        # for each prediction, update the csv file.
        current_exp_path = os.path.join(out_dir, exp_names[i].decode("utf-8"))
        paths.create_dir(out_dir)
        paths.create_dir(current_exp_path)

        for j in range(len(labels)):
            # filename = "%03d_predict_%s.csv" % (j, labels[j])
            filename = "predict_%s.csv" % labels[j]
            current_exp_file = os.path.join(current_exp_path,
                                            filename)
            with open(current_exp_file, "w") as outfile:
                write_csv(outfile,
                          labels[j],
                          predictions[:, i, j],
                          ground_truth[i][:, :, j],
                          frames[i])


def write_predictions3(out_dir, exp_names, predictions, ground_truth, frames):
    """Write predictions to disk."""
    labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    for i in range(len(exp_names)):
        # for each prediction, update the csv file.
        current_exp_path = os.path.join(out_dir, exp_names[i])
        paths.create_dir(out_dir)
        paths.create_dir(current_exp_path)

        for j in range(len(labels)):
            # filename = "%03d_predict_%s.csv" % (j, labels[j])
            filename = "predict_%s.csv" % labels[j]
            current_exp_file = os.path.join(current_exp_path,
                                            filename)

            with open(current_exp_file, "w") as outfile:
                write_csv(outfile,
                          labels[j],
                          predictions[i][:, j],
                          ground_truth[i][:, j],
                          frames[i])


def make_sequence_predictions(network, videos):
    """Given a network, make predictions.

    Inputs -
      network: The network dictionary. Assumes that there is a predict_seq
        key. This dictionary entry will do the full sequence predictions"
      videos: A python list of videos.
    Outputs -
      predictions: A python list of the network outputs. (List instead of an
        array due to the arbitrary sequence lengths).
    """
    predictions = []
    # for i in range(len(videos)):
    for i in range(videos.shape[1]):
        predictions.append(network["predict_seq"](videos[i])[0])
    return predictions


def make_sequence_predictions2(network, videos):
    """Given a network, make predictions.

    Inputs -
      network: The network dictionary. Assumes that there is a predict_seq
        key. This dictionary entry will do the full sequence predictions"
      videos: A python list of videos.
    Outputs -
      predictions: A python list of the network outputs. (List instead of an
        array due to the arbitrary sequence lengths).
    """
    predictions = []
    hs = []
    cs = []
    # for i in range(len(videos)):
    for i in range(videos.shape[1]):
        predict, h, c = network["predict_seq"](videos[:, i, :])
        predictions.append(predict)
        hs.append(h)
        cs.append(c)
    return predictions, hs, cs


def calculate_grad_stats(network):
    """Compute graident stats."""
    num_grads = len(network["grad_vals"])
    grad_stats = numpy.zeros((num_grads, 5))
    t = network["lr_update"]["params"][0]

    for j in range(num_grads):
        g = network["grad_vals"][j]
        grad_stats[j][0] = t.get_value()
        grad_stats[j][1] = g.get_value().mean()
        grad_stats[j][2] = g.get_value().max()
        grad_stats[j][3] = g.get_value().min()
        grad_stats[j][4] = g.get_value().std()
        # # print "\t%s: %f" % (g.name, g.get_value().mean())
        # filename = "%s/grads/%s.html" % (base_filename, g.name)
        # data = grad_stats[j][0:plot_idx + 1, :]
        # plot_helper.line_plot(data, names=["mean", "max", "min", "std"],
        #                       title=g.name, filename=filename)
    return grad_stats


def copy_outputs(opts, train_data, valid_data):
    """Copy outputs (images and templates)."""
    shutil.copy("templates/graph.html", opts["out_dir"])
    shutil.copy("templates/main.js", opts["out_dir"])
    shutil.copy("templates/require.js", opts["out_dir"])

    if opts["image_dir"] is not None:
        for i in range(len(train_data["exp"])):
            # copy images
            input_dir = os.path.join(
                opts["image_dir"], train_data["exp"][i][0], "frames")
            base_out = os.path.join(
                opts["out_dir"], "predictions", "train",
                train_data["exp"][i][0])
            output_dir = os.path.join(base_out, "frames")
            paths.create_dir(base_out)

            if platform.system() == "Linux":
                if not os.path.exists(output_dir):
                    os.symlink(input_dir, output_dir)
            # else:
            #     shutil.copytree(input_dir, output_dir)
            # copy templates
            copy_templates(base_out)

        for i in range(len(valid_data["exp"])):
            input_dir = os.path.join(
                opts["image_dir"], valid_data["exp"][i][0], "frames")
            base_out = os.path.join(
                opts["out_dir"], "predictions", "valid",
                valid_data["exp"][i][0])
            output_dir = os.path.join(base_out, "frames")
            paths.create_dir(base_out)

            if platform.system() == "Linux":
                if not os.path.exists(output_dir):
                    os.symlink(input_dir, output_dir)
            # else:
            #     shutil.copytree(input_dir, output_dir)
            # copy templates
            copy_templates(base_out)

    return


def copy_outputs2(opts, train_data, valid_data):
    """Copy outputs (images and templates)."""
    shutil.copy("templates/graph.html", opts["out_dir"])
    shutil.copy("templates/main.js", opts["out_dir"])
    shutil.copy("templates/require.js", opts["out_dir"])

    if opts["image_dir"] is not None:
        for i in range(len(train_data["exp"])):
            # copy images
            input_dir = os.path.join(
                opts["image_dir"], train_data["exp"][i], "frames")
            base_out = os.path.join(
                opts["out_dir"], "predictions", "train",
                train_data["exp"][i])
            output_dir = os.path.join(base_out, "frames")
            paths.create_dir(base_out)

            if platform.system() == "Linux":
                if not os.path.exists(output_dir):
                    os.symlink(input_dir, output_dir)
            # else:
            #     shutil.copytree(input_dir, output_dir)
            # copy templates
            copy_templates(base_out)

        for i in range(len(valid_data["exp"])):
            input_dir = os.path.join(
                opts["image_dir"], valid_data["exp"][i], "frames")
            base_out = os.path.join(
                opts["out_dir"], "predictions", "valid",
                valid_data["exp"][i])
            output_dir = os.path.join(base_out, "frames")
            paths.create_dir(base_out)

            if platform.system() == "Linux":
                if not os.path.exists(output_dir):
                    os.symlink(input_dir, output_dir)
            # else:
            #     shutil.copytree(input_dir, output_dir)
            # copy templates
            copy_templates(base_out)

    return


def copy_main_graphs(opts, out_dir=None):
    """Copy templates."""
    if out_dir is None:
        if "out_dir" in list(opts.keys()):
            out_dir = opts["out_dir"]
        else:
            out_dir = opts["flags"].out_dir

    out_dir = os.path.join(out_dir, "plots")
    shutil.copy("templates/graph.html", out_dir)
    shutil.copy("templates/main.js", out_dir)
    shutil.copy("templates/require.js", out_dir)

    create_loss_plot(out_dir, "templates/plot_csv.html", "f_score.csv")
    create_loss_plot(out_dir, "templates/plot_csv.html", "loss.csv")
    shutil.copy("templates/plot_csv.js", out_dir)
    # shutil.copy("templates/loss_f.html", out_dir)
    # shutil.copy("templates/loss_f.js", out_dir)

    # shutil.copy("templates/loss_f_curve.html", out_dir)
    # shutil.copy("templates/loss_f_curve.js", out_dir)

    return


def copy_experiment_graphs(opts, all_exp_dir, exp_names):
    """Create space for mouse video/experiment outputs."""
    if "image_dir" in list(opts.keys()):
        image_dir = opts["image_dir"]
    else:
        image_dir = opts["flags"].image_dir

    for i in range(len(exp_names)):
        # copy images
        base_out = os.path.join(
            all_exp_dir,
            exp_names[i].decode("utf-8"))
        paths.create_dir(base_out)

        if image_dir is not None:
            input_dir = os.path.join(
                image_dir, exp_names[i].decode("utf-8"), "frames")
            output_dir = os.path.join(base_out, "frames")

            if platform.system() == "Linux":
                # if not os.path.exists(output_dir):
                # need a check for broken symlinks.
                if not os.path.lexists(output_dir):
                    os.symlink(input_dir, output_dir)
        # else:
        #     shutil.copytree(input_dir, output_dir)
        # copy templates
        copy_templates2(base_out)
    return


def copy_movie_graphs(opts, all_exp_dir, h5_data, label_names, fps=30):
    """Copy movie playing templates."""
    exp_names = h5_data["exp_names"][()]
    proc_labels = ["lift", "hand", "grab", "suppinate", "mouth", "chew"]
    copy_proc = False
    for label_name in label_names:
        if label_name in proc_labels:
            copy_proc = True
            break

    for i in range(len(exp_names)):
        # create the output path
        exp = h5_data["exps"][exp_names[i]]
        base_out = os.path.join(all_exp_dir, exp_names[i].decode("utf-8"))
        paths.create_dir(base_out)
        video_name = os.path.basename(exp["video_name"][()].decode("utf-8"))

        if copy_proc is True:
            shutil.copy("templates/processed_main.js", base_out)

        # copy the movie file
        movie_file = os.path.join(
            opts["flags"].display_dir, exp["video_name"][()].decode("utf-8"))
        out_movie = os.path.join(
            base_out, video_name
        )
        if platform.system() == "Linux":
            if not os.path.lexists(out_movie):
                os.symlink(movie_file, out_movie)

        # copy/create the html/js templates
        shutil.copy("templates/require.js", base_out)
        shutil.copy("templates/movie_viewer.js", base_out)
        # for each label_name
        for label_name in label_names:
            csv_name = "predict_" + label_name + ".csv"
            create_html_file(
                base_out, csv_name, video_name, fps
            )
            # csv_name = "processed_" + label_name + ".csv"
            # create_html_file(
            #     base_out, csv_name, video_name, fps
            # )
            if label_name in proc_labels:
                # copy over processed
                proc_name = "processed_" + label_name
                shutil.copy(
                    "templates/%s.html" % proc_name,
                    base_out)


def create_html_file(out_dir, csv_name, video_name, fps):
    in_filename = os.path.join("templates/predict_movie_template.html")
    out_filename = os.path.join(out_dir, csv_name[:-3] + "html")

    with open(in_filename, "r") as template:
        with open(out_filename, "w") as out_file:
            keyval_dict = {
                 "csv": '"' + csv_name + '"',
                 "movie": '"' + video_name + '"',
                 "fps": fps
            }
            for line in template:
                new_line = template_helper.parse_line(line, keyval_dict)
                out_file.write(new_line)

def create_loss_plot(out_dir, template_file, csv_name):
    in_filename = os.path.join(template_file)
    out_filename = os.path.join(out_dir, csv_name[:-3] + "html")

    with open(in_filename, "r") as template:
        with open(out_filename, "w") as out_file:
            keyval_dict = {
                 "csv": '"' + csv_name + '"',
            }
            for line in template:
                new_line = template_helper.parse_line(line, keyval_dict)
                out_file.write(new_line)


def write_predictions_list(out_dir, exp_names, predictions, ground_truth,
                           hidden_list, frames, label_names):
    """Write predictions to disk."""
    for i in range(len(exp_names)):
        # for each prediction, update the csv file.
        current_exp_path = os.path.join(out_dir, exp_names[i].decode("utf-8"))
        paths.create_dir(out_dir)
        paths.create_dir(current_exp_path)

        for j in range(len(label_names)):
            # filename = "%03d_predict_%s.csv" % (j, labels[j])
            filename = "predict_%s.csv" % label_names[j]
            current_exp_file = os.path.join(current_exp_path,
                                            filename)
            with open(current_exp_file, "w") as outfile:
                write_csv(outfile,
                          label_names[j],
                          predictions[:, i, j],
                          ground_truth[i][:, :, j],
                          frames[i])


def log_outputs2(opts, step, train_cost, test_cost, label_names,
                 frame_thresh=[10, 10, 10, 10, 10, 10]):
    # apply post processing (hungarian matching and create cleaned outputs).
    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "train")
    train_dicts = post_processing.process_outputs2(
        predict_dir, "", label_names, frame_thresh=frame_thresh)

    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "test")
    test_dicts = post_processing.process_outputs2(
        predict_dir, "", label_names, frame_thresh=frame_thresh)

    # after applying the post processing,
    trainf, trainf_scores = compute_tpfp(opts, train_dicts)
    testf, testf_scores = compute_tpfp(opts, test_dicts)

    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss_f.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,train f1,test f1,"
                     "\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = ("%d,%f,%f,%f,%f,"
                      "\n")
        output_data =\
            [step, train_cost, test_cost, trainf, testf]
        output_data = tuple(output_data)
        outfile.write(format_str % output_data)
    print("\tupdated...")


def log_outputs3(opts, step, train_cost, test_cost, valid_cost, label_names,
                 frame_thresh=[10, 10, 10, 10, 10, 10]):
    # apply post processing (hungarian matching and create cleaned outputs).
    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "train")
    train_dicts = post_processing.process_outputs2(
        predict_dir, "", label_names, frame_thresh=frame_thresh)

    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "test")
    test_dicts = post_processing.process_outputs2(
        predict_dir, "", label_names, frame_thresh=frame_thresh)

    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "valid")
    valid_dicts = post_processing.process_outputs2(
        predict_dir, "", label_names, frame_thresh=frame_thresh)

    # after applying the post processing,
    trainf, trainf_scores = compute_tpfp(opts, train_dicts)
    testf, testf_scores = compute_tpfp(opts, test_dicts)
    validf, valid_scores = compute_tpfp(opts, valid_dicts)

    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss_f.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,valid cost,"
                     "train f1,test f1,valid f1\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = ("%d,%f,%f,%f,%f,%f,%f"
                      "\n")
        output_data =\
            [step, train_cost, test_cost, valid_cost, trainf, testf, validf]
        output_data = tuple(output_data)
        outfile.write(format_str % output_data)
    print("\tupdated...")


def compute_tpfp(opts, label_dicts):
    """Compute the precision recall information."""
    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    mean_f = 0
    for i in range(len(label_dicts)):
        tp = float(len(label_dicts[i]['dists']))
        fp = float(label_dicts[i]['fp'])
        fn = float(label_dicts[i]['fn'])
        precision = tp / (tp + fp + opts["eps"])
        recall = tp / (tp + fn + opts["eps"])
        f1_score =\
            2 * (precision * recall) / (precision + recall + opts["eps"])

        total_tp += tp
        total_fp += fp
        total_fn += fn
        # print "label: %s" % label_dicts[i]['label']
        # print "\tprecision: %f" % precision
        # print "\trecall: %f" % recall
        # print "\tfscore: %f" % f1_score
        # mean_f += f1_score
        f1_scores.append(f1_score)

    precision = total_tp / (total_tp + total_fp + opts["eps"])
    recall = total_tp / (total_tp + total_fn + opts["eps"])
    mean_f = 2 * (precision * recall) / (precision + recall + opts["eps"])
    # mean_f = (mean_f / len(label_dicts))
    # print "mean score: %f" % mean_f

    return mean_f, f1_scores


def copy_templates(opts, data, type_name, label_names):
    copy_main_graphs(opts)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", type_name)
    if not os.path.exists(base_out):
        os.makedirs(base_out)

    copy_movie_graphs(
        opts, base_out, data, label_names
    )


def copy_templates2(out_dir):
    """Copy templates to output directory."""
    shutil.copy("templates/require.js", out_dir)
    shutil.copy("templates/predict_main.js", out_dir)
    # copy templates
    shutil.copy("templates/predict_lift.html", out_dir)
    # shutil.copy("templates/predict_lift.js", out_dir)

    shutil.copy("templates/predict_hand.html", out_dir)
    # shutil.copy("templates/predict_hand.js", out_dir)

    shutil.copy("templates/predict_grab.html", out_dir)
    # shutil.copy("templates/predict_grab.js", out_dir)

    shutil.copy("templates/predict_suppinate.html", out_dir)
    # shutil.copy("templates/predict_suppinate.js", out_dir)

    shutil.copy("templates/predict_mouth.html", out_dir)
    # shutil.copy("templates/predict_mouth.js", out_dir)

    shutil.copy("templates/predict_chew.html", out_dir)
    # shutil.copy("templates/predict_chew.js", out_dir)
