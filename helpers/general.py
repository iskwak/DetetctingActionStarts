import helpers.sequences_helper as sequences_helper
import helpers.post_processing as post_processing
import numpy as np
import h5py
from torch.autograd import Variable
import helpers.arg_parsing as arg_parsing
import gflags
import os


def setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the number iterations per epoch.
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        num_train_vids = len(train_data["exp_names"])
        iter_per_epoch =\
            np.ceil(1.0 * num_train_vids / opts["flags"].hantman_mini_batch)

        iter_per_epoch = int(iter_per_epoch)
        opts["flags"].iter_per_epoch = iter_per_epoch
        opts["flags"].total_iterations =\
            iter_per_epoch * opts["flags"].total_epochs

    return opts


def copy_templates(opts, train_data, test_data):
    print("copying frames/templates...")
    sequences_helper.copy_main_graphs(opts)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "train")
    # train_experiments = exp_list[train_vids]
    train_experiments = train_data["exp_names"].value
    sequences_helper.copy_experiment_graphs(
        opts, base_out, train_experiments)

    base_out = os.path.join(opts["flags"].out_dir, "predictions", "test")
    # test_experiments = exp_list[train_vids]
    test_experiments = test_data["exp_names"].value
    sequences_helper.copy_experiment_graphs(
        opts, base_out, test_experiments)


def create_match_array(opts, net_out, org_labels, label_weight):
    """Create the match array."""
    val_threshold = 0.7
    frame_threshold = 10
    y_org = org_labels

    COST_FP = 20
    # COST_FN = 20
    net_out = net_out.data.cpu().numpy()
    num_frames, num_vids, num_classes = net_out.shape
    TP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    FP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    num_false_neg = []
    num_false_pos = []
    for i in range(num_vids):
        temp_false_neg = 0
        temp_false_pos = 0
        for j in range(num_classes):
            processed, max_vals = post_processing.nonmax_suppress(
                net_out[:, i, j], val_threshold)
            processed = processed.reshape((processed.shape[0], 1))
            data = np.zeros((len(processed), 3), dtype="float32")
            data[:, 0] = list(range(len(processed)))
            data[:, 1] = processed[:, 0]
            data[:, 2] = y_org[:, i, j]
            # if opts["flags"].debug is True:
            #     import pdb; pdb.set_trace()
            # after suppression, apply hungarian.
            labelled = np.argwhere(y_org[:, i, j] == 1)
            labelled = labelled.flatten().tolist()
            num_labelled = len(labelled)
            dist_mat = post_processing.create_frame_dists(
                data, max_vals, labelled)
            rows, cols, dist_mat = post_processing.apply_hungarian(
                dist_mat, frame_threshold)

            # missed classifications
            # false_neg = len(labelled) - len(
            #     [k for k in range(len(max_vals)) if cols[k] < len(labelled)])
            # num_false_neg += false_neg
            # temp_false_neg += false_neg

            num_matched = 0
            for pos in range(len(max_vals)):
                ref_idx = max_vals[pos]
                # if cols[pos] < len(labelled):
                row_idx = rows[pos]
                col_idx = cols[pos]
                if col_idx < len(labelled) and\
                        dist_mat[row_idx, col_idx] < frame_threshold:
                    # True positive
                    label_idx = labelled[cols[pos]]
                    TP_weight[ref_idx, i, j] = np.abs(ref_idx - label_idx)
                    # import pdb; pdb.set_trace()
                    # if we are reweighting based off label rariety
                    if opts["flags"].reweight is True:
                        TP_weight[ref_idx, i, j] =\
                            TP_weight[ref_idx, i, j] * label_weight[j]
                    num_matched += 1
                else:
                    # False positive
                    FP_weight[ref_idx, i, j] = COST_FP
                    if opts["flags"].reweight is True:
                        FP_weight[ref_idx, i, j] =\
                            FP_weight[ref_idx, i, j] * label_weight[j]
                    temp_false_pos += 1

            temp_false_neg += num_labelled - num_matched
        num_false_neg.append(temp_false_neg)
        num_false_pos.append(temp_false_pos)

    num_false_neg = np.asarray(num_false_neg).astype("float32")
    return TP_weight, FP_weight, num_false_neg, num_false_pos


def get_label_weight(opts, data):
    """Get number of positive examples for each label."""
    experiments = data["exp_names"].value
    label_mat = np.zeros((experiments.size, 7))
    vid_lengths = np.zeros((experiments.size,))

    for i in range(experiments.size):
        exp_key = experiments[i]
        exp = data["exps"][exp_key]
        for j in range(6):
            # label_counts[j] += exp["org_labels"].value[:, j].sum()
            label_mat[i, j] = exp["org_labels"].value[:, j].sum()
        # label_counts[-1] +=\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()

        # label_mat[i, -1] =\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = exp["org_labels"].shape[0]

    # this is a weight on the counts. 
    label_weight = 1.0 / (6 * np.mean(label_mat, axis=0) + opts["eps"])
    label_weight = label_weight

    label_weight[-1] = 1.0
    # batch_weight = 1 / (opts["flags"].hantman_mini_batch / 2.0)

    # label_weight = label_weight * batch_weight
    # import pdb; pdb.set_trace()
    # label_weight[-2] = label_weight[-2] * 10
    if opts["flags"].reweight is False:
        label_weight = np.asarray([1, 1, 1, 1, 1, 1, .01])

    return label_weight


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


def log_outputs(opts, step, train_cost, test_cost):
    # apply post processing (hungarian matching and create cleaned outputs).
    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "train")
    train_dicts = post_processing.process_outputs(
        predict_dir, "")

    predict_dir = os.path.join(opts["flags"].out_dir,
                               "predictions", "test")
    test_dicts = post_processing.process_outputs(
        predict_dir, "")

    # after applying the post processing,
    trainf, trainf_scores = compute_tpfp(opts, train_dicts)
    testf, testf_scores = compute_tpfp(opts, test_dicts)

    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss_f.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,train f1,test f1,"
                     "train lift,train hand,train grab,train supinate,"
                     "train mouth,train chew,"
                     "test lift,test hand,test grab,test supinate,"
                     "test mouth,test chew\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = ("%d,%f,%f,%f,%f,"
                      "%f,%f,%f,%f,"
                      "%f,%f,"
                      "%f,%f,%f,%f,"
                      "%f,%f\n")
        output_data =\
            [step, train_cost, test_cost, trainf, testf] +\
            trainf_scores + testf_scores
        output_data = tuple(output_data)
        # import pdb; pdb.set_trace()
        outfile.write(format_str % output_data)
    print("\tupdated...")


def eval_network(opts, step, network, label_weight, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()

        num_frames = inputs[0].size()[0]
        chunk_len = 50
        num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
        predict = np.zeros((num_frames, 1, 6))
        # print(num_frames)
        for j in range(0, num_chunks):
            idx1 = j * chunk_len
            idx2 = min((j + 1) * chunk_len, num_frames)
            # print("\t%d" % idx2)
            chunk = [
                Variable(inputs[0][idx1:idx2, :, :, :], requires_grad=True).cuda(),
                Variable(inputs[1][idx1:idx2, :, :, :], requires_grad=True).cuda(),
                Variable(inputs[2][idx1:idx2, :], requires_grad=False).cuda()
            ]
            out = network([chunk[0], chunk[1]])

            predict[idx1:idx2, 0, :] = out.data.cpu().numpy()
            loss = criterion(out, chunk[-1])
            total_loss += loss.data[0]
        exp_names = [inputs[3]]
        labels = inputs[2].cpu().numpy()
        labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        frames = [range(labels[0].shape[0])]
        sequences_helper.write_predictions2(out_dir, exp_names, predict, labels,
                                            None, frames)
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch)
