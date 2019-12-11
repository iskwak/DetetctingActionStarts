"""Mouse behavior spike classification."""
from __future__ import print_function, division
import os
import time
import sys
import gflags
import numpy as np

import h5py
import helpers.paths as paths
import helpers.arg_parsing as arg_parsing

import helpers.sequences_helper as sequences_helper
import helpers.post_processing as post_processing
# import models.hantman_hungarian as hantman_hungarian
from models import mpii_model as hantman_hungarian
import flags.lstm_flags
import flags.cuda_flags
import torch
from helpers.RunningStats import RunningStats

# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("image_dir", None, "Directory for images to symlink.")
gflags.DEFINE_integer("total_iterations", 0,
                      "Don't set for this version of the training code.")
# gflags.DEFINE_boolean("debug", False, "Debug flag, work with less videos.")
gflags.DEFINE_integer("update_iterations", None,
                      "Number of iterations to output logging information.")
gflags.DEFINE_integer("iter_per_epoch", None,
                      "Number of iterations per epoch. Leave empty.")
gflags.DEFINE_integer("save_iterations", None,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))
gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")
gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_boolean("threaded", True, "Threaded Data loadered.")
gflags.DEFINE_boolean("anneal", True, "Use annealing on perframe cost.")
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_list("feat_keys", None, "Feature keys to use.")
gflags.DEFINE_string("arch", "concat", "Which lstm arch to use.")
# gflags.DEFINE_float(
#     "hantman_weight_decay", 0.0001, "Weight decay value.")

gflags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
gflags.DEFINE_integer(
    "hantman_mini_batch", 256,
    "Mini batch size for training.")
# gflags.DEFINE_integer("hantman_seq_length", 1500, "Sequence length.")
gflags.DEFINE_boolean("normalize", False, "Normalize data.")


gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("train_file")
gflags.MarkFlagAsRequired("test_file")
gflags.MarkFlagAsRequired("feat_keys")
# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(hantman_hungarian)
gflags.ADOPT_module_key_flags(flags.lstm_flags)
gflags.ADOPT_module_key_flags(flags.cuda_flags)


# label_idx = [72, 25, 0, 84, 8, 58, 43, 37, 61, 77, 15]
# label_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# label_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
label_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13]


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)

    # setup the number iterations per epoch.
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        num_train_vids = len(train_data["exp_names"])
        iter_per_epoch =\
            np.ceil(1.0 * num_train_vids / opts["flags"].hantman_mini_batch)

        iter_per_epoch = int(iter_per_epoch)
        print(iter_per_epoch)
        opts["flags"].iter_per_epoch = iter_per_epoch
        opts["flags"].total_iterations =\
            iter_per_epoch * opts["flags"].total_epochs

    return opts


def _init_network(opts, h5_data, label_weight):
    """Setup the network."""
    exp_list = h5_data["exp_names"].value
    # import pdb; pdb.set_trace()
    opts["feat_dims"] = [
        train_data["exps"][exp_list[0]][feat_key].shape[2]
        for feat_key in opts["flags"].feat_keys
    ]
    # num_input = h5_data["exps"][exp_list[0]]["reduced"].shape[2]
    # num_classes = h5_data["exps"][exp_list[0]]["labels"].shape[2]
    num_classes = len(label_idx)
    if opts["flags"].arch == "concat":
        network = hantman_hungarian.HantmanHungarianConcat(
            input_dims=opts["feat_dims"],
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=num_classes,
            label_weight=label_weight
        )
        # network = torch.nn.DataParallel(network, device_ids=[0, 2])
    elif opts["flags"].arch == "sum":
        network = hantman_hungarian.HantmanHungarianSum(
            input_dims=opts["feat_dims"],
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=num_classes,
            label_weight=label_weight
        )
    else:
        network = hantman_hungarian.HantmanHungarianBidirConcat(
            input_dims=opts["feat_dims"],
            hidden_dim=opts["flags"].lstm_hidden_dim,
            output_dim=num_classes,
            label_weight=label_weight
        )

    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    if opts["flags"].cuda_device != -1:
        network.cuda()

    return network, optimizer


def _copy_templates(opts, train_data, test_data):
    print("copying frames/templates...")
    sequences_helper.copy_main_graphs(opts)

    # base_out = os.path.join(opts["flags"].out_dir, "predictions", "train")
    # # train_experiments = exp_list[train_vids]
    # train_experiments = train_data["exp_names"].value
    # sequences_helper.copy_experiment_graphs(
    #     opts, base_out, train_experiments)

    # base_out = os.path.join(opts["flags"].out_dir, "predictions", "test")
    # # test_experiments = exp_list[train_vids]
    # test_experiments = test_data["exp_names"].value
    # sequences_helper.copy_experiment_graphs(
    #     opts, base_out, test_experiments)


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
                    # if opts["flags"].reweight is True:
                    #     TP_weight[ref_idx, i, j] =\
                    #         TP_weight[ref_idx, i, j] * label_weight[j]
                    num_matched += 1
                else:
                    # False positive
                    FP_weight[ref_idx, i, j] = COST_FP
                    # if opts["flags"].reweight is True:
                    #     FP_weight[ref_idx, i, j] =\
                    #         FP_weight[ref_idx, i, j] * label_weight[j]
                    temp_false_pos += 1

            temp_false_neg += num_labelled - num_matched
        num_false_neg.append(temp_false_neg)
        num_false_pos.append(temp_false_pos)

    num_false_neg = np.asarray(num_false_neg).astype("float32")
    return TP_weight, FP_weight, num_false_neg, num_false_pos


def compute_tpfp(label_dicts):
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


def _get_feat(opts, feat, seq_len=None):
    if seq_len is None:
        seq_len = np.min([opts["flags"].seq_len, feat.shape[0]])

    temp_feat = feat.reshape((feat.shape[0], feat.shape[2]))

    return temp_feat, seq_len


def _log_outputs(opts, step, network, label_weight):
    """Log the outputs for the network."""
    # Run the network on all the training and testing examples.
    # Creates a graph for each video.
    train_cost, train_scores = _process_full_sequences(
        opts, step, network, train_data, "train", label_weight)
    test_cost, test_scores = _process_full_sequences(
        opts, step, network, test_data, "test", label_weight)

    # apply post processing (hungarian matching and create cleaned outputs).
    # predict_dir = os.path.join(opts["flags"].out_dir,
    #                            "predictions", "train")
    # train_dicts = post_processing.process_outputs(
    #     predict_dir, "")

    # predict_dir = os.path.join(opts["flags"].out_dir,
    #                            "predictions", "test")
    # test_dicts = post_processing.process_outputs(
    #     predict_dir, "")

    # after applying the post processing,
    # trainf, trainf_scores = compute_tpfp(train_dicts)
    # testf, testf_scores = compute_tpfp(test_dicts)

    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss_f.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,"
                     "train tp,train fp,train fn,train perframe,"
                     "test tp,test fp,test fn,test perframe\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = ("%d,%f,%f,"
                      "%f,%f,%f,%f,"
                      "%f,%f,%f,%f\n")
        output_data =\
            [step, train_cost, test_cost] +\
            train_scores + test_scores
        output_data = tuple(output_data)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        outfile.write(format_str % output_data)
    print("\tupdated...")


def _get_hidden(opts):
    if opts["flags"].cuda_device >= 0:
        use_cuda = True
    else:
        use_cuda = False

    hidden = network.init_hidden(
        opts["flags"].hantman_mini_batch,
        use_cuda=use_cuda)
    return hidden


def _train_epoch(opts, step, network, optimizer, train_data, test_data, label_weight):
    """Train one epoch."""
    # loss_fp_fn = os.path.join(opts["flags"].out_dir, "plots", "loss_fp.csv")
    train_exps = train_data["exp_names"].value
    train_exps = opts["rng"].permutation(train_exps)
    round_tic = time.time()
    batch_id = 0

    # inputs, org_labels, sample_idx, batch_id = _get_seq_mini_batch(
    #     opts, batch_id, train_data, train_exps)
    while batch_id != -1:
        # inputs, org_labels, sample_idx, batch_id = _get_seq_mini_batch(
        #     opts, batch_id, train_data, train_exps)
        inputs, labels, mask, org_labels, sample_idx, batch_id =\
            _get_seq_mini_batch(opts, batch_id, train_data, train_exps)
        # train_predict = network["predict_batch"](inputs[0])
        hidden = _get_hidden(opts)
        # img_side = torch.autograd.Variable(torch.Tensor(inputs[0])).cuda()
        # img_front = torch.autograd.Variable(torch.Tensor(inputs[1])).cuda()
        inputs = [
            torch.autograd.Variable(torch.Tensor(feats), requires_grad=True).cuda()
            for feats in inputs
        ]
        labels = torch.autograd.Variable(torch.Tensor(labels), requires_grad=False).cuda()
        mask = torch.autograd.Variable(torch.Tensor(mask), requires_grad=False).cuda()

        train_predict, update_hid = network(inputs, hidden)

        TP_weight, FP_weight, false_neg, false_pos = create_match_array(
            opts, train_predict, org_labels, label_weight[2])

        pos_mask, neg_mask = hantman_hungarian.create_pos_neg_masks(labels, label_weight[0], label_weight[1])
        perframe_cost = hantman_hungarian.perframe_loss(train_predict, mask, labels, pos_mask, neg_mask)
        tp_cost, fp_cost, fn_cost = hantman_hungarian.structured_loss(
            train_predict, mask, TP_weight, FP_weight, false_neg)
        if step % 10 == 0:
            import pdb; pdb.set_trace()
        total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost =\
            hantman_hungarian.combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost)
        # import pdb; pdb.set_trace()
        cost = total_cost.mean()
        optimizer.zero_grad()
        cost.backward()
        # torch.nn.utils.clip_grad_norm(network.parameters(), 5)

        optimizer.step()
        step += 1
        if step % 1000 == 0:
            print(cost.data)
    return step


def _train_network(opts, network, optimizer, train_data, test_data, label_weight):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    step = 0
    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        network.train()
        step = _train_epoch(opts, step, network, optimizer, train_data, test_data, label_weight)
        print("\tFinished epoch")
        print("\tProcessing all examples...")
        network.eval()
        # _log_outputs(opts, step, network, label_weight)
        round_tic = time.time()

        # # save the network in its own folder in the networks folder
        out_dir = os.path.join(
            opts["flags"].out_dir, "networks", "%d" % step)
        paths.create_dir(out_dir)
        out_name = os.path.join(out_dir, "network.pt")
        torch.save(network.cpu().state_dict(), out_name)
        network.cuda()
        # hantman_hungarian_image.save_network(opts, network, out_name)
    print("Finished training.")


def _process_full_sequences(opts, step, network, h5_data, name, label_weight):
    """Make per sequence predictions."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    idx = h5_data["exp_names"].value
    return _predict_write(opts, step, out_dir, network, h5_data, idx, label_weight)


def _predict_write(opts, step, out_dir, network, h5_data, exps, label_weight):
    """Predict and write sequence classifications."""
    batch_id = 0
    exps.sort()
    loss = 0
    scores = [0, 0, 0, 0]
    batch_count = 0
    # t = network["lr_update"]["params"][0]
    t = step
    while batch_id != -1:
        if batch_count % 10 == 0:
            print("\t\t%d" % batch_count)
        # inputs, org_labels, sample_idx, batch_id = _get_seq_mini_batch(
        #     opts, batch_id, h5_data, exps)
        inputs, labels, mask, org_labels, sample_idx, batch_id =\
            _get_seq_mini_batch(opts, batch_id, h5_data, exps)

        hidden = _get_hidden(opts)
        # img_side = torch.autograd.Variable(torch.Tensor(inputs[0])).cuda()
        # img_front = torch.autograd.Variable(torch.Tensor(inputs[1])).cuda()
        inputs = [
            torch.autograd.Variable(torch.Tensor(feats)).cuda()
            for feats in inputs
        ]
        labels = torch.autograd.Variable(torch.Tensor(labels)).cuda()
        mask = torch.autograd.Variable(torch.Tensor(mask)).cuda()

        predict, update_hid = network(inputs, hidden)

        TP_weight, FP_weight, false_neg, false_pos = create_match_array(
            opts, predict, org_labels, label_weight[2])

        pos_mask, neg_mask = hantman_hungarian.create_pos_neg_masks(labels, label_weight[0], label_weight[1])
        perframe_cost = hantman_hungarian.perframe_loss(predict, mask, labels, pos_mask, neg_mask)
        tp_cost, fp_cost, fn_cost = hantman_hungarian.structured_loss(
            predict, mask, TP_weight, FP_weight, false_neg)

        total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost =\
            hantman_hungarian.combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost)
        cost = total_cost.mean()

        loss += cost.data[0]
        # order from past:
        # total cost, struct_cost, tp, fp, fn, perframe
        scores[0] += tp_cost.data.cpu()[0]
        scores[1] += fp_cost.data.cpu()[0]
        scores[2] += fn_cost.data.cpu()[0]
        scores[3] += perframe_cost.data.cpu()[0]

        # scores = [scores[i] + cost[i + 3] for i in range(len(cost[3:]))]
        predictions = predict.data.cpu().numpy()

        # collect the labels
        labels = []
        frames = []
        for vid in exps[sample_idx]:
            labels.append(h5_data["exps"][vid]["labels"].value)
            frames.append(list(range(h5_data["exps"][vid]["labels"].shape[0])))

        # idx = idx[:valid_idx]
        # print feat_idx
        # print idx
        # print inputs
        # import pdb; pdb.set_trace()
        # sequences_helper.write_predictions2(
        #     out_dir, exps[sample_idx], predictions, labels,
        #     [], frames)

        batch_count = batch_count + 1

    loss = loss / batch_count
    scores = [score / batch_count for score in scores]

    return loss, scores


def _get_seq_mini_batch(opts, batch_id, h5_data, idx):
    """Get a mini-batch of data."""
    if "mini_batch_size" in list(opts.keys()):
        batch_size = opts["mini_batch_size"]
    else:
        batch_size = opts["flags"].hantman_mini_batch

    start_idx = batch_id * batch_size
    end_idx = start_idx + batch_size
    if end_idx >= len(idx):
        # valid_idx = end_idx - len(idx)
        buf = [0 for i in range(end_idx - len(idx))]
        end_idx = len(idx) - 1
        sample_idx = np.asarray(list(range(start_idx, len(idx))) + buf,
                                dtype="int64")
        vid_idx = sample_idx

        # valid_idx = np.max(np.argwhere(vid_idx + 1 == len(idx))) + 1
        batch_id = -1
    else:
        sample_idx = np.asarray(list(range(start_idx, end_idx)), dtype="int64")
        vid_idx = sample_idx

        batch_id += 1
        # valid_idx = len(sample_idx)
    # import pdb; pdb.set_trace()
    # feat_dims = h5_data["exps"][idx[0]]["img_side"].shape[2]
    feat_dims = opts["feat_dims"]

    all_feats = [
        np.zeros((opts["flags"].seq_len, batch_size, feat_dim),
                 dtype="float32")
        for feat_dim in feat_dims
    ]

    all_labels = np.zeros((opts["flags"].seq_len, batch_size, len(label_idx)),
                          dtype="float32")

    all_masks = np.zeros((opts["flags"].seq_len, batch_size, len(label_idx)),
                         dtype="float32")
    all_org_labels = np.zeros((opts["flags"].seq_len, batch_size, len(label_idx)),
                              dtype="float32")

    for i in range(batch_size):
        temp_idx = idx[vid_idx[i]]

        seq_len = min(
            opts["flags"].seq_len,
            h5_data["exps"][temp_idx]["labels"].shape[0])
        for j in range(len(feat_dims)):
            feat_key = opts["flags"].feat_keys[j]

            temp_feat, _ = _get_feat(
                opts, h5_data["exps"][temp_idx][feat_key].value, seq_len=seq_len)
            # all_feats[j][:seq_len, i, :] = temp_feat[:seq_len, :]
            # ... need to clean up the code
            # abuse the globals...
            temp_feat = (temp_feat[:seq_len, :] - means[j]) / stds[j]
            # try:
            all_feats[j][:seq_len, i, :] = temp_feat[:seq_len, :]
            # except:
            #     import pdb; pdb.set_trace()

        label = h5_data["exps"][temp_idx]["labels"].value[:, :, label_idx]
        temp_label = label.reshape((label.shape[0], label.shape[2]))
        all_labels[:seq_len, i, :] = temp_label[:seq_len, :]

        org_labels = h5_data["exps"][temp_idx]["org_labels"].value[:, label_idx]
        all_org_labels[:seq_len, i, :] = org_labels[:seq_len, :]

        all_masks[:seq_len, i, :] = 1
        # all_masks[i, :, :] = h5_data["exps"][temp_idx]["mask"].value

    # import pdb; pdb.set_trace()

    # return func_inputs, all_org_labels, vid_idx, batch_id
    return all_feats, all_labels, all_masks, all_org_labels, vid_idx, batch_id


def _get_label_weight(data):
    """Get number of positive examples for each label."""
    experiments = data["exp_names"].value
    label_mat = np.zeros((experiments.size, len(label_idx) + 1))
    label_mat2 = np.zeros((experiments.size, len(label_idx) + 1))
    vid_lengths = np.zeros((experiments.size,))
    seq_len = opts["flags"].seq_len

    for i in range(experiments.size):
        exp_key = experiments[i]
        exp = data["exps"][exp_key]
        for j in range(len(label_idx)):
            # label_counts[j] += exp["org_labels"].value[:, j].sum()
            curr_idx = label_idx[j]
            label_mat[i, j] = exp["org_labels"].value[:, curr_idx].sum()
            label_mat2[i, j] = exp["org_labels"].value[:seq_len, curr_idx].sum()
        # label_counts[-1] +=\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()
        label_mat[i, -1] =\
            exp["org_labels"].shape[0] - exp["org_labels"].value[:, label_idx].sum()

        label_mat2[i, -1] =\
            seq_len - exp["org_labels"].value[:seq_len, label_idx].sum()
        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = exp["org_labels"].shape[0]

    # label_counts = label_mat.sum(axis=0)
    # label_weight = 1.0 / np.mean(label_mat, axis=0)
    # import pdb; pdb.set_trace()
    label_weight = 1.0 / np.mean(label_mat2, axis=0)
    label_weight[:-1] = label_weight[:-1] * 100

    # label_weight[-2] = label_weight[-2] * 10
    if opts["flags"].reweight is False:
        label_weight = [5, 5, 5, 5, 5, 5, .01]
    # import pdb; pdb.set_trace()
    return label_weight


def compute_means(opts, train_data):
    """Go over the features and compute the mean and variance."""
    exp_names = train_data["exp_names"].value
    means = []
    stds = []
    if opts["flags"].normalize is True:
        running_stats = []
        for key in opts["flags"].feat_keys:
            temp_feat = train_data["exps"][exp_names[0]][key].value
            running_stats.append(
                RunningStats(temp_feat.shape[2])
            )
        # loop over the experiments

        # for exp_name in exp_names:
        # import pdb; pdb.set_trace()
        for j in range(len(exp_names)):
            exp_name = exp_names[j]
            print(exp_name)
            # loop over the keys
            for i in range(len(opts["flags"].feat_keys)):
                key = opts["flags"].feat_keys[i]

                seq_len = train_data["exps"][exp_name]["labels"].shape[0]
                print("\t%d" % seq_len)
                # figure out last label location
                # temp_labels = train_data["exps"][exp_name]["org_labels"].value[:, label_idx]
                # print(np.argwhere(temp_labels)[-1])
                # import pdb; pdb.set_trace()
                temp_feat = train_data["exps"][exp_name][key].value
                temp_feat = temp_feat[:seq_len, 0, :]

                running_stats[i].add_data(
                    temp_feat
                )

        for i in range(len(opts["flags"].feat_keys)):
            means.append(running_stats[i].mean)
            stds.append(running_stats[i].compute_std())
    else:
        for key in opts["flags"].feat_keys:
            temp_feat = train_data["exps"][exp_names[0]][key].value
            mean = np.zeros((temp_feat.shape[2], ))
            std = np.ones((temp_feat.shape[2], ))
            means.append(mean)
            stds.append(std)

    return means, stds


if __name__ == "__main__":
    print(sys.argv)

    opts = _setup_opts(sys.argv)
    paths.setup_output_space(opts)
    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    # load data
    # try:
    tic = time.time()
    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            _copy_templates(opts, train_data, test_data)
            label_weight = _get_label_weight(train_data)
            label_mat = np.tile(
                label_weight,
                (opts["flags"].seq_len,
                    opts["flags"].hantman_mini_batch, 1)).astype('float32')
            pos_weight = torch.Tensor(label_mat[:, :, :-1]).cuda()
            neg_weight = torch.Tensor([float(label_mat[0, 0, -1])]).cuda()
            label_weight = [pos_weight, neg_weight, label_weight]
            network, optimizer = _init_network(opts, train_data, label_weight)

            print("a")
            means, stds = compute_means(opts, train_data)
            print("b")
            _train_network(opts, network, optimizer, train_data, test_data, label_weight)
    print("moo")
    temp = time.time() - tic
    out_file = os.path.join(opts["flags"].out_dir, "timing.txt")
    with open(out_file, "w") as fd:
        fd.write("%f\n" % temp)
    # except Exception as e:
    #     with open(os.path.join(opts["flags"].out_dir, "errors.txt"), "w") as fd:
    #         fd.write(str(e))
    #         print(str(e))
