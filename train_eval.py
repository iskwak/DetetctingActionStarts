"""Helper to have some training functions."""
from __future__ import print_function, division
import numpy
import time
import gflags
import torch
from scipy import signal
import os
import helpers.paths as paths
from helpers.RunningStats import RunningStats
import helpers.sequences_helper as sequences_helper
import helpers.post_processing as post_processing
import helpers.hungarian_matching as hungarian_matching

gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("valid_file", None, "Valid data filename (hdf5).")

gflags.DEFINE_integer("total_epochs", 500, "Total number of epochs.")
gflags.DEFINE_integer("save_epochs", 10, "Epoch interval to save the network.")
gflags.DEFINE_integer(
    "update_iterations", 10,
    "Epoch interval to update network perforamce. Processes all the data, can be slow.")
gflags.DEFINE_integer("save_iterations", 10,
                      ("Number of iterations to save the network (expensive "
                       "to do this)."))

gflags.DEFINE_string("load_network", None, "Cached network to load.")
gflags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
gflags.DEFINE_integer("mini_batch", 256, "Mini batch size for training.")
gflags.DEFINE_list("feat_keys", None, "Feature keys to use.")

gflags.DEFINE_string("display_dir", None, "Directory of videos for display.")
gflags.DEFINE_string(
    "video_dir", None,
    "Directory for processing videos, (codecs might be different from display)")

gflags.DEFINE_integer(
    "label_smooth_win", 19,
    "Window size for label smoothing. Use 0 for no smoothing.")
gflags.DEFINE_integer(
    "label_smooth_std", 2,
    "Window size for label smoothing. Use 0 for no smoothing.")

gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_boolean("normalize", True, "Normalize the inputs.")


def get_label_weight(train_sampler):
    """Get number of positive examples for each label."""
    # experiments = data["exp_names"].value
    experiments = train_sampler.exp_names
    label_dims = train_sampler.label_dims

    # the number of labels is only the "positive" behaviors. The
    # do nothing label needs to be included.
    label_mat = numpy.zeros((experiments.size, label_dims + 1))
    vid_lengths = numpy.zeros((experiments.size,))
    train_sampler.reset()

    # ... hack... turn off gpu for a bit...

    for i in range(train_sampler.num_batch):
        blob = train_sampler.get_minibatch()
        num_vids = blob["labels"].shape[1]
        # the seq_len is 1 plus the last 1 entry in the mask.
        masks = blob["masks"].cpu().numpy()
        labels = blob["labels"].cpu().numpy()

        # get the seq lens
        seq_len = 0
        for j in range(num_vids):
            seq_len += numpy.max(numpy.argwhere(masks[:, j, :])) + 1

        # for each label/behavior, compute the number of appearances
        for j in range(label_dims):
            label_mat[i, j] += labels[:, :, j].sum()
            # for k in range(num_vids):
            #     label_mat[i, j] += labels[:, k, j].sum()

        label_mat[i, -1] =\
            seq_len - labels.sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = seq_len

    # label_counts = label_mat.sum(axis=0)

    label_weight = 1.0 / numpy.mean(label_mat, axis=0)
    # protect from div 0 issues.
    label_weight[numpy.isinf(label_weight)] = 0

    return label_weight


def train_lstm(opts, network, optimizer, criterion, samplers, fid):
    """Trains the lstm network.

    Given opts, network, optimizer, criterion and samplers, this function will
    train a network given the hyper parameters provided in opts.

    It will save the network to disk every opts["flags"].save_iterations times,
    and will save the network outputs to disk every
    opts["flags"].update_iterations times.
    """
    # normalize the features
    # tic = time.time()
    print("computing means")
    epoch_tic = time.time()
    tic = time.time()
    # running_stats, preproc_feat = compute_means(opts, samplers[1])
    running_stats, preproc_feat = compute_means(opts, samplers[0])
    toc = time.time()
    fid.write("means,%f\n" % (toc - tic))
    fid.flush()
    # print(time.time() - tic)
    # update the sampelrs with the feature preprocessing and label smoothing

    def preproc_label(labels):
        return smooth_data(opts, labels)

    for i in range(len(samplers)):
        if samplers[i] is not None:
            # samplers[i].feat_pre = preproc_feat
            samplers[i].feat_pre = None
            samplers[i].label_pre = preproc_label

    train_eval = samplers[0]
    test_eval = samplers[1]
    valid_eval = None # samplers[3]
    frame_thresh = [10, 10, 10, 10, 10, 10]
    timings = [0, 0, 0, 0]

    step = numpy.floor(opts["flags"].iter_per_epoch) * opts["flags"].total_epochs

    # one last evaluation
    network.eval()
    tic = time.time()
    train_loss, train_match, test_loss, test_match, valid_loss, valid_match =\
        eval_network(opts, step, network, criterion, train_eval,
                        test_eval, valid_eval, frame_thresh=frame_thresh)
    # write to disk
    write_loss_scores(opts, step, train_loss, test_loss, valid_loss)
    write_f_scores(opts, step, train_match, test_match, valid_match)
    toc = time.time()
    fid.write("eval network,%f\n" % (toc - tic))

    # save the network in its own folder in the networks folder
    tic = time.time()
    out_dir = os.path.join(
        opts["flags"].out_dir, "networks", "%d" % step)
    paths.create_dir(out_dir)
    out_name = os.path.join(out_dir, "network.pt")
    torch.save(network.cpu().state_dict(), out_name)
    toc = time.time()
    fid.write("save network,%f\n" % (toc - tic))
    network.cuda()



def write_f_scores(opts, step, train_match, test_match, valid_match):
    # write to the graph.
    fscore_fname = os.path.join(opts["flags"].out_dir, "plots", "f_score.csv")
    if os.path.isfile(fscore_fname) is False:
        with open(fscore_fname, "w") as f:
            f.write("iteration,training fscore,test fscore,valid fscore")
            for i in range(len(train_match)):
                f.write(",train %s,test %s,valid %s" %
                        (train_match[i]["labels"], test_match[i]["labels"],
                         valid_match[i]["labels"]))
            f.write("\n")
    with open(fscore_fname, "a") as outfile:
        train_all, train_class = compute_fscores(opts, train_match)
        test_all, test_class = compute_fscores(opts, test_match)
        valid_all, valid_class = compute_fscores(opts, valid_match)

        # write out the data...
        format_str = "%d,%f,%f,%f"
        output_data =\
            [step, train_all, test_all, valid_all]

        for i in range(len(train_match)):
            format_str += ",%f,%f,%f"
            output_data += [train_class[i], test_class[i], valid_class[i]]

        output_data = tuple(output_data)
        outfile.write(format_str % output_data)
        outfile.write("\n")


def compute_fscores(opts, label_dicts):
    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    mean_f = 0

    for i in range(len(label_dicts)):
        tp = float(label_dicts[i]['tps'])
        fp = float(label_dicts[i]['fps'])
        fn = float(label_dicts[i]['fns'])
        precision = tp / (tp + fp + opts["eps"])
        recall = tp / (tp + fn + opts["eps"])
        f1_score =\
            2 * (precision * recall) / (precision + recall + opts["eps"])

        total_tp += tp
        total_fp += fp
        total_fn += fn
        f1_scores.append(f1_score)

    precision = total_tp / (total_tp + total_fp + opts["eps"])
    recall = total_tp / (total_tp + total_fn + opts["eps"])
    mean_f = 2 * (precision * recall) / (precision + recall + opts["eps"])

    return mean_f, f1_scores


def write_loss_scores(opts, step, train_loss, test_loss, valid_loss):
    # write to the graph.
    loss_f = os.path.join(opts["flags"].out_dir, "plots", "loss.csv")
    if os.path.isfile(loss_f) is False:
        with open(loss_f, "w") as f:
            f.write(("iteration,training loss,test loss,valid loss\n"))
    with open(loss_f, "a") as outfile:
        # write out the data...
        format_str = "%d,%f,%f,%f\n"
        output_data =\
            [step, train_loss, test_loss, valid_loss]
        output_data = tuple(output_data)
        outfile.write(format_str % output_data)


def eval_network(opts, step, network, criterion, train_eval, test_eval,
                 valid_eval, frame_thresh=[10, 10, 10, 10, 10, 10]):
    # train_loss, train_match = process_seqs(
    #     opts, step, network, train_eval, criterion, "train")
    test_loss, test_match = process_seqs(
        opts, step, network, test_eval, criterion, "test")
    if valid_eval is not None:
        valid_loss, valid_match = process_seqs(
            opts, step, network, valid_eval, criterion, "valid")
    else:
        valid_loss = test_loss
        valid_match = test_match
    train_loss = test_loss
    train_match = test_match

    return train_loss, train_match, test_loss, test_match, valid_loss, valid_match


def process_seqs(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0
    if sampler.batch_idx.empty():
        sampler.reset()

    label_names = sampler.label_names
    # init dict
    all_matches = []
    for i in range(len(label_names)):
        all_matches.append({
            "tps": 0,
            "fps": 0,
            "fns": 0,
            "labels": label_names[i]
        })
    for i in range(sampler.num_batch):
        blob = sampler.get_minibatch()

        if name == "test":
            hidden = get_hidden(opts, network, batchsize=1)
        else:
            hidden = get_hidden(opts, network)
        mask = blob["masks"]
        # hidden = (hidden[0][:, :1, :].clone(), hidden[1][:, :1, :].clone())

        predict, update_hid = network(
            blob["features"], hidden)
        # conv_labels = smooth_data(opts, blob["labels"])

        labels = blob["labels"]
        labels = labels * mask
        labels.cuda()
        conv_labels = blob["proc_labels"]
        conv_labels = conv_labels * mask
        conv_labels = conv_labels.cuda()

        predict = predict * mask
        pos_mask, neg_mask = create_pos_neg_masks(conv_labels)
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()

        # cost = criterion(step, predict, conv_labels, pos_mask, neg_mask, mask)
        cost = criterion(step, labels, conv_labels, predict, pos_mask, neg_mask, mask)
        exp_names = blob["names"]
        # labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        labels = blob["labels"]
        labels = [labels]
        frames = [range(labels[0].shape[0])]
        for j in range(len(exp_names)):
            temp_labels = [labels[0][:, j:j+1, :]]
            sequences_helper.write_predictions2(
                out_dir, [exp_names[j]], predict[:, j:j+1, :],
                temp_labels, None, frames, label_names=label_names)

            # process ouputs
            matches = analyze_outs(out_dir, [exp_names[j]],
                                   predict[:, j:j+1, :], temp_labels,
                                   label_names)
            # merge the values
            all_matches = merge_dicts(all_matches, matches)

        total_loss += cost.item()
        # leaking mem...?
        torch.cuda.empty_cache()

    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch), all_matches


def merge_dicts(all_matches, matches):
    for i in range(len(matches)):
        all_matches[i]["tps"] += len(matches[i]["tps"])
        all_matches[i]["fps"] += len(matches[i]["fps"])
        all_matches[i]["fns"] += matches[i]["num_fn"]
    return all_matches


def analyze_outs(out_dir, exp_names, predict, labels, label_names, frame_thresh=0.7):
    # next apply non max suppression
    labels = labels[0]
    num_labels = labels.shape[2]

    all_matches = []
    for i in range(num_labels):
        ground_truth = labels[:, 0, i]
        gt_sup, gt_idx = post_processing.nonmax_suppress(
            ground_truth, frame_thresh)
        predict_sup, predict_idx = post_processing.nonmax_suppress(
            predict[:, 0, i], frame_thresh)
        match_dict, dist_mat = hungarian_matching.apply_hungarian(
            gt_idx, predict_idx
        )

        # write processed file
        output_name = os.path.join(
            out_dir, exp_names[0], 'processed_%s.csv' % label_names[i]
        )
        create_proc_file(output_name, gt_sup, predict_sup, match_dict)
        all_matches.append(match_dict)

    return all_matches


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


def check_tps(frame_num, tps_list):
    # Helper function to find if a there is a match for the current frame.
    for sample in tps_list:
        if frame_num == sample[1]:
            return sample[0]
    return -1


def compute_means(opts, train_sampler):
    """Go over the features and compute the mean and variance."""

    running_stats = []
    for dim in train_sampler.feat_dims:
        running_stats.append(
            RunningStats(dim)
        )
    if opts["flags"].normalize is True:
        # loop over the experiments
        train_sampler.reset()

        max_val = 0
        for i in range(train_sampler.num_batch):
            # loop over the keys
            blob = train_sampler.get_minibatch()
            num_vids = blob["features"][0].shape[1]
            masks = blob["masks"].cpu().numpy()
            for j in range(len(blob["features"])):
                seq_len = numpy.max(numpy.argwhere(masks[:, j, :])) + 1
                feats = blob["features"][j].cpu().detach().numpy()
                for k in range(num_vids):
                    temp_feat = feats[:seq_len, k, :]

                    running_stats[j].add_data(
                        temp_feat
                    )
                    if temp_feat.max() > max_val:
                        max_val = temp_feat.max()
    # else: if no normalize, the initialization of RunningStats is already
    # 0 mean, 1 std/var.

    # construct the preprocessing function.
    feat_keys = list(zip(
        range(len(opts["flags"].feat_keys)),
        opts["flags"].feat_keys
    ))
    def preproc_feats(features, feat_key):
        # key_idx = list(filter(
        #     lambda key: feat_key == key[1], feat_keys
        # ))
        for i in range(len(feat_keys)):
            if feat_keys[i][1] == feat_key:
                key_idx = feat_keys[i][0]
                break
        # length of key_idx should be 1.
        # key_idx = key_idx[0][0]
        means = running_stats[key_idx].mean
        stds = running_stats[key_idx].compute_std()

        features = ((features - means) / stds).astype("float32")
        return features

    return running_stats, preproc_feats


def train_lstm_epoch(opts, step, network, optimizer, criterion, sampler):
    """Train one epoch."""
    if sampler.batch_idx.empty():
        sampler.reset()

    timings = [0, 0, 0, 0]
    for i in range(sampler.num_batch):
        # get data blob
        tic = time.time()
        blob = sampler.get_minibatch()
        hidden = get_hidden(opts, network)

        # prepare data for network
        # inputs = []
        # for j in range(len(blob["features"])):
        #     # temp = ((blob["features"][j] - means[j])/stds[j]).astype("float32")
        #     # temp = blob["features"][j].astype("float32")
        #     # inputs.append(torch.tensor(temp, requires_grad=True).cuda())
        #     inputs.append(blob["features"][j])
        # # labels = torch.tensor(blob["labels"], requires_grad=False).cuda()
        # frame_mask = torch.tensor(blob["masks"], requires_grad=False).cuda()
        frame_mask = blob["masks"]
        conv_labels = blob["proc_labels"]
        labels = blob["labels"]
        inputs = blob["features"]
        timings[0] = timings[0] + (time.time() - tic)

        # run the network
        tic = time.time()
        train_predict, update_hid = network(inputs, hidden)
        timings[1] = timings[1] + (time.time() - tic)
        temp = train_predict.cpu().detach().numpy()
        # if numpy.argwhere(numpy.isnan(temp)).size > 0:
        #     import pdb; pdb.set_trace()

        # prepare for criterion
        tic = time.time()
        # conv_labels = smooth_data(opts, blob["labels"])
        conv_labels = conv_labels * frame_mask
        train_predict = train_predict * frame_mask

        conv_labels = torch.tensor(conv_labels, requires_grad=False).cuda()
        # conv_labels = conv_labels.clone().detach().requires_grad_(False).cuda()

        pos_mask, neg_mask = create_pos_neg_masks(conv_labels)
        timings[2] = timings[2] + (time.time() - tic)

        # apply criterion and backprop.
        tic = time.time()
        cost = criterion(step, labels, conv_labels, train_predict, pos_mask, neg_mask, frame_mask)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        timings[3] = timings[3] + (time.time() - tic)
        step += 1
        # print(cost)

    timings = [
        timing / sampler.num_batch for timing in timings
    ]
    print(timings)
    return step, timings


def create_pos_neg_masks(labels):
    """Create pos/neg masks."""
    temp = labels.data
    pos_mask = (temp > 0.7).float()
    neg_mask = (temp <= 0.7).float()

    return pos_mask, neg_mask


def get_hidden(opts, network, batchsize=None):
    if opts["flags"].cuda_device >= 0:
        use_cuda = True
    else:
        use_cuda = False
    if batchsize == None:
        batchsize = opts["flags"].mini_batch

    hidden = network.init_hidden(
        batchsize,
        use_cuda=use_cuda)
    return hidden


def smooth_data(opts, org_labels):
    """Apply gaussian smoothing to the labels."""
    # smooth_window = 19
    # smooth_std = 2
    smooth_window = opts["flags"].label_smooth_win
    smooth_std = opts["flags"].label_smooth_std

    if smooth_window == 0:
        return org_labels

    # org_labels = labels
    labels = numpy.zeros(org_labels.shape, dtype="float32")
    # loop over the columns and convolve
    conv_filter = signal.gaussian(smooth_window, std=smooth_std)
    for i in range(labels.shape[1]):
        labels[:, i] = numpy.convolve(
            org_labels[:, i], conv_filter, 'same')

    # scale the labels a bit
    # labels = labels * 0.9
    # labels = labels + 0.01
    # plot the labels somewhere.

    # test_dir = os.path.join(opts["flags"].out_dir, "test")
    # if not os.path.exists(test_dir):
    #     os.mkdir(test_dir)

    # # debug... lets just write out the first example.
    # out_name = os.path.join(test_dir, "data.csv")
    # num_labels = org_labels.shape[2]
    # with open(out_name, "w") as fid:
    #     fid.write("frame")
    #     for i in range(num_labels):
    #         fid.write(", behav %d" % i)
    #     fid.write("\n")

    #     for i in range(0, labels.shape[0]):
    #         fid.write("%f" % i)
    #         for j in range(0, num_labels):
    #             fid.write(", %f" % labels[i, 1, j])
    #         fid.write("\n")
    return labels


# def smooth_data(opts, org_labels):
#     """Apply gaussian smoothing to the labels."""
#     # smooth_window = 19
#     # smooth_std = 2
#     smooth_window = opts["flags"].label_smooth_win
#     smooth_std = opts["flags"].label_smooth_std

#     if smooth_window == 0:
#         return org_labels

#     # org_labels = labels
#     labels = numpy.zeros(org_labels.shape, dtype="float32")
#     # loop over the columns and convolve
#     conv_filter = signal.gaussian(smooth_window, std=smooth_std)

#     for i in range(labels.shape[1]):
#         for j in range(labels.shape[2]):
#             labels[:, i, j] = numpy.convolve(
#                 org_labels[:, i, j], conv_filter, 'same')

#     # scale the labels a bit
#     # labels = labels * 0.9
#     # labels = labels + 0.01
#     # plot the labels somewhere.

#     # test_dir = os.path.join(opts["flags"].out_dir, "test")
#     # if not os.path.exists(test_dir):
#     #     os.mkdir(test_dir)

#     # # debug... lets just write out the first example.
#     # out_name = os.path.join(test_dir, "data.csv")
#     # num_labels = org_labels.shape[2]
#     # with open(out_name, "w") as fid:
#     #     fid.write("frame")
#     #     for i in range(num_labels):
#     #         fid.write(", behav %d" % i)
#     #     fid.write("\n")

#     #     for i in range(0, labels.shape[0]):
#     #         fid.write("%f" % i)
#     #         for j in range(0, num_labels):
#     #             fid.write(", %f" % labels[i, 1, j])
#     #         fid.write("\n")
#     # import pdb; pdb.set_trace()
#     return labels
