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
import flags.cuda_flags
import torch
from models.hantman_3dconv import Hantman3DConv
from helpers.videosampler import HantmanVideoFrameSampler
from helpers.videosampler import HantmanVideoSampler
from torch.autograd import Variable
import helpers.hungarian_matching as hungarian_matching

DEBUG = False
# flags for processing hantman files.
gflags.DEFINE_string("out_dir", None, "Output directory path.")
gflags.DEFINE_string("train_file", None, "Train data filename (hdf5).")
gflags.DEFINE_string("test_file", None, "Test data filename (hdf5).")
gflags.DEFINE_string("valid_file", None, "Valid data filename (hdf5).")
gflags.DEFINE_string("display_dir", None, "Directory of videos for display.")
gflags.DEFINE_string(
    "video_dir", None,
    "Directory for processing videos, (codecs might be different from display)")
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
gflags.DEFINE_boolean("reweight", True, "Try re-weighting.")
gflags.DEFINE_string(
    "frames",
    "-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10",
    "Frames to process.")
# gflags.DEFINE_float(
#     "hantman_weight_decay", 0.0001, "Weight decay value.")

gflags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
gflags.DEFINE_integer(
    "hantman_mini_batch", 256,
    "Mini batch size for training.")
# needed to limit how much to process during eval. important for speed purposes.
gflags.DEFINE_integer("seq_len", 1500, "Sequence length.")

gflags.MarkFlagAsRequired("out_dir")
gflags.MarkFlagAsRequired("train_file")
gflags.MarkFlagAsRequired("test_file")

# gflags.DEFINE_boolean("help", False, "Help")
gflags.ADOPT_module_key_flags(arg_parsing)
gflags.ADOPT_module_key_flags(flags.cuda_flags)

g_label_names = [
    "lift", "hand", "grab", "supinate", "mouth", "chew"
]


def _setup_opts(argv):
    """Parse inputs."""
    FLAGS = gflags.FLAGS
    opts = arg_parsing.setup_opts(argv, FLAGS)
    # this is dumb... passing in negative numbers to DEFINE_multi_int doesn't
    # seem to work well. So frames will be a string and split off of spaces.
    opts["flags"].frames = [
        int(frame_num) for frame_num in opts["flags"].frames.split(' ')
    ]

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


def _create_labels(opts, label_mat):
    label_idx = torch.LongTensor(label_mat.size(0))

    # pos_idx = label_mat.data.nonzero()
    # label_idx[:pos_idx.shape[0]] = pos_idx[:, 1]

    pos_idx = label_mat.nonzero()
    label_idx[:pos_idx.shape[0]] = pos_idx.data[:, 1]
    label_idx[pos_idx.shape[0]:] = 6

    if opts["flags"].cuda_device != -1:
        label_idx = label_idx.cuda()

    # label_idx = Variable(label_idx, requires_grad=False)

    return label_idx


def _init_network(opts, label_weight):
    """Setup the network."""
    network = Hantman3DConv().cuda()
    # create the optimizer too
    optimizer = torch.optim.Adam(
        network.parameters(), lr=opts["flags"].learning_rate)

    label_weight = torch.Tensor(label_weight)
    criterion = torch.nn.NLLLoss(label_weight)

    if opts["flags"].cuda_device != -1:
        network.cuda()
        criterion.cuda()

    return network, optimizer, criterion


def _train_epoch(opts, network, optimizer, criterion, sampler):
    sampler.reset()
    sampling = []
    gpuing = []

    for i in range(sampler.num_batch):
        # print("%d of %d" % (i, sampler.num_batch))
        tic = time.time()
        blob = sampler.get_minibatch()

        label_idx = _create_labels(opts, blob[2])
        sampling.append(time.time() - tic)

        tic = time.time()
        out, _ = network(blob[:2])

        loss = criterion(out, label_idx)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        gpuing.append(time.time() - tic)
        # torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()

    print("\t%f" % np.mean(sampling))
    print("\t%f" % np.mean(gpuing))
    # print("\t%f" % np.min(sampling))

    return sampler.num_batch


def _train_network(opts, network, optimizer, criterion,
                   sampler, train_eval, test_eval, valid_eval):
    """Train the network."""
    print("Beginning training...")
    # train_exps = train_data["experiments"].value
    # train_exps.sort()
    frame_thresh = [
        10 for label in g_label_names
    ]
    step = 0

    for i in range(opts["flags"].total_epochs):
        print("EPOCH %d, %d" % (i, step))
        tic = time.time()
        network.train()
        step += _train_epoch(opts, network, optimizer, criterion, sampler)
        print("\t%f" % (time.time() - tic))
        print("\tFinished epoch")

        # if DEBUG and i % 20 == 0 and i != 0:
        # if i % 20 == 0 and i != 0:
        # if i % 20 == 0 and i != 0:
        if i % opts["flags"].update_iterations == 0 and i != 0:
            print("\tProcessing all examples...")
            tic = time.time()
            network.eval()
            # train_loss, test_loss, valid_loss = _eval_network(
            #     opts, step, network, criterion, train_eval, test_eval, valid_eval)
            train_loss, train_match, test_loss, test_match, valid_loss, valid_match =\
                _eval_network(opts, step, network, criterion, train_eval,
                            test_eval, valid_eval, frame_thresh=frame_thresh)

            _write_loss_scores(opts, step, train_loss, test_loss, valid_loss)
            _write_f_scores(opts, step, train_match, test_match, valid_match)
            # sequences_helper.log_outputs3(
            #     opts, step, train_cost, test_cost, valid_cost, g_label_names,
            #     frame_thresh=frame_thresh)
            # import pdb; pdb.set_trace()

            # save the network in its own folder in the networks folder
            out_dir = os.path.join(
                opts["flags"].out_dir, "networks", "%d" % step)
            paths.create_dir(out_dir)
            out_name = os.path.join(out_dir, "network.pt")
            torch.save(network.cpu().state_dict(), out_name)
            network.cuda()
            print("\tProcessing finished: %f" % (time.time() - tic))

        network.eval()
        # train_loss, test_loss, valid_loss = _eval_network(
        #     opts, step, network, criterion, train_eval, test_eval, valid_eval)
        train_loss, train_match, test_loss, test_match, valid_loss, valid_match =\
            _eval_network(opts, step, network, criterion, train_eval,
                        test_eval, valid_eval, frame_thresh=frame_thresh)

        _write_loss_scores(opts, step, train_loss, test_loss, valid_loss)
        _write_f_scores(opts, step, train_match, test_match, valid_match)
        # save the network in its own folder in the networks folder
        out_dir = os.path.join(
            opts["flags"].out_dir, "networks", "%d" % step)
        paths.create_dir(out_dir)
        out_name = os.path.join(out_dir, "network.pt")
        torch.save(network.cpu().state_dict(), out_name)
        network.cuda()

    print("Finished training.")


def _write_loss_scores(opts, step, train_loss, test_loss, valid_loss):
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
        # import pdb; pdb.set_trace()
        outfile.write(format_str % output_data)


def _write_f_scores(opts, step, train_match, test_match, valid_match):
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
        # import pdb; pdb.set_trace()
        outfile.write(format_str % output_data)
        outfile.write("\n")


def _eval_network(opts, step, network, criterion, train_eval, test_eval,
                  valid_eval, frame_thresh=[10, 10, 10, 10, 10, 10]):
    # train_loss, train_match = _process_seqs(opts, step, network, train_eval,
    train_loss, train_match = _process_seqs(opts, step, network, train_eval, criterion, "train")
    if DEBUG is False:
        test_loss, test_match = _process_seqs(opts, step, network, test_eval, criterion, "test")
        valid_loss, valid_match = _process_seqs(opts, step, network, valid_eval, criterion, "valid")
    else:
        test_loss, test_match = train_loss, train_match
        valid_loss, valid_match = train_loss, train_match

    return train_loss, train_match, test_loss, test_match, valid_loss, valid_match


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


def _process_seqs(opts, step, network, sampler, criterion, name):
    """Evaluate the state of the network."""
    out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
    total_loss = 0
    if sampler.batch_idx.empty():
        sampler.reset()

    tic = time.time()
    num_behav = len(g_label_names)
    # init dict
    all_matches = []
    for i in range(len(g_label_names)):
        all_matches.append({
            "tps": 0,
            "fps": 0,
            "fns": 0,
            "labels": g_label_names[i]
        })
    for i in range(sampler.num_batch):
        inputs = sampler.get_minibatch()
        # num_frames = inputs[0].size()[0]
        #  ; pdb.set_trace()
        num_frames = np.min([opts["flags"].seq_len, inputs[0].size()[0]])
        # chunk_len = 4
        # print(inputs[3][0].decode("utf-8"))
        # print("\t%d" % num_frames)

        # num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
        predict = np.zeros((num_frames, 1, num_behav))
        # print(num_frames)
        for j in range(0, num_frames):

            feat1, feat2, labels = _create_chunks(opts, inputs, j)
            chunk = [feat1, feat2, labels]

            if opts["flags"].cuda_device != -1:
                chunk = [
                    feat1.cuda(), feat2.cuda(), labels.cuda()
                ]

            # idx1 = j * chunk_len
            # idx2 = min((j + 1) * chunk_len, num_frames)
            # # print("\t%d" % idx2)
            # chunk = [
            #     Variable(inputs[0][idx1:idx2, :1, :, :], requires_grad=True).cuda(),
            #     Variable(inputs[1][idx1:idx2, :1, :, :], requires_grad=True).cuda(),
            #     Variable(inputs[2][idx1:idx2, :], requires_grad=False).cuda()
            # ]
            out, features = network([chunk[0], chunk[1]])
            # import pdb; pdb.set_trace()
            out = torch.exp(out.data)

            predict[j, 0, :] = out[:, :num_behav].data.cpu().numpy()
            # label_idx = _create_labels(opts, chunk[-1])

            loss = criterion(out, chunk[-1])
            total_loss += loss.item()

        exp_names = [inputs[3][0]]
        labels = inputs[2].cpu().numpy()
        labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
        frames = [range(num_frames)]
        sequences_helper.write_predictions2(
            out_dir, exp_names, predict, labels, None, frames,
            label_names=g_label_names
        )

        # process ouputs
        matches = analyze_outs(out_dir, exp_names, predict, labels)
        # merge the values
        all_matches = _merge_dicts(all_matches, matches)
        # # add label names
        # for j in range(len(g_label_names)):
        #     all_matches[j]["label"] = g_label_names[j]

        # if len(all_matches[0]["tps"]) > 0:
        #     import pdb; pdb.set_trace()

    print("\t%f" % (time.time() - tic))
    print("\teval loss: %f" % (total_loss / sampler.num_batch))
    return (total_loss / sampler.num_batch), all_matches


def _merge_dicts(all_matches, matches):
    for i in range(len(matches)):
        all_matches[i]["tps"] += len(matches[i]["tps"])
        all_matches[i]["fps"] += len(matches[i]["fps"])
        all_matches[i]["fns"] += matches[i]["num_fn"]
    return all_matches


# def _eval_network(opts, step, network, sampler, criterion, name):
#     """Evaluate the state of the network."""
#     out_dir = os.path.join(opts["flags"].out_dir, "predictions", name)
#     total_loss = 0

#     tic = time.time()
#     num_behav = len(g_label_names)
#     if sampler.batch_idx.empty():
#         sampler.reset()
#     for i in range(sampler.num_batch):
#         inputs = sampler.get_minibatch()
#         # num_frames = inputs[0].size()[0]
#         #  ; pdb.set_trace()
#         num_frames = np.min([opts["flags"].seq_len, inputs[0].size()[0]])
#         # chunk_len = 4
#         print(inputs[3][0].decode("utf-8"))
#         print("\t%d" % num_frames)

#         # num_chunks = int(np.ceil(1.0 * num_frames / chunk_len))
#         predict = np.zeros((num_frames, 1, num_behav))
#         # print(num_frames)
#         for j in range(0, num_frames):
#             feat1, feat2, labels = _create_chunks(opts, inputs, j)
#             chunk = [feat1, feat2, labels]

#             if opts["flags"].cuda_device != -1:
#                 chunk = [
#                     feat1.cuda(), feat2.cuda(), labels.cuda()
#                 ]

#             out, _ = network(chunk[:2])
#             # if len((labels != 6).nonzero().size()):
#             #     import pdb; pdb.set_trace()
#             temp = torch.exp(out.data)

#             predict[j, 0, :] = temp[:, :num_behav].cpu().numpy()

#             loss = criterion(out, chunk[-1])
#             total_loss += loss.item()
#             # total_loss += loss.data[0]

#         exp_names = [inputs[3][0]]
#         labels = inputs[2].cpu().numpy()
#         labels = [labels.reshape((labels.shape[0], 1, labels.shape[1]))]
#         frames = [range(labels[0].shape[0])]
#         # import pdb; pdb.set_trace()
#         sequences_helper.write_predictions_list(out_dir, exp_names, predict,
#                                                 labels, None, frames,
#                                                 g_label_names)
#         matches = analyze_outs(out_dir, exp_names, predict, labels)
#         import pdb; pdb.set_trace()
#     print(time.time() - tic)
#     print("\teval loss: %f" % (total_loss / sampler.num_batch))
#     return (total_loss / sampler.num_batch)


def analyze_outs(out_dir, exp_names, predict, labels, frame_thresh=0.7):
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
            out_dir, exp_names[0].decode("utf-8"), 'processed_%s.csv' % g_label_names[i]
        )
        create_proc_file(output_name, gt_sup, predict_sup, match_dict)
        all_matches.append(match_dict)
        # import pdb; pdb.set_trace()
    return all_matches


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


def _load_chunk(opts, inputs, frames):
    feat1 = torch.zeros(1, 1, len(frames), 224, 224)
    feat2 = torch.zeros(1, 1, len(frames), 224, 224)

    if np.any(np.array(frames) < 0):
        for j in range(len(frames)):
            if frames[j] < 0:
                frames[j] = 0
    if np.any(np.array(frames) >= inputs[0].size(0)):
        for j in range(len(frames)):
            if frames[j] >= inputs[0].size(0):
                frames[j] = inputs[0].size(0) - 1
    idxs = range(0, 10)
    for i, frame in zip(idxs, frames):
        feat1[:, 0, i, :, :] = inputs[0][frame, 0, :, :]
        feat2[:, 0, i, :, :] = inputs[1][frame, 0, :, :]
    # import pdb; pdb.set_trace()
    return feat1, feat2


def _create_chunks(opts, inputs, frame_idx):
    """Create the overlapping chunks."""
    # idx2 = 75
    # idx1 = 71
    # num_batch = idx2 - idx1
    # img1 = torch.zeros(num_batch, 1, 10, 224, 224)
    # img2 = torch.zeros(num_batch, 1, 10, 224, 224)
    # labels = torch.zeros(num_batch)

    feat1_list = []
    feat2_list = []
    label_list = []

    frames = [
        offset + frame_idx for offset in opts["flags"].frames
    ]
    temp1, temp2 = _load_chunk(opts, inputs, frames)
    feat1_list.append(temp1)
    feat2_list.append(temp2)

    temp_label = inputs[2][frame_idx, :].nonzero()
    if len(temp_label.size()) == 1:
        temp_label = 6
    else:
        temp_label = temp_label[0][0]
    label_list.append(temp_label)

    feat1 = torch.cat(feat1_list, dim=0)
    feat2 = torch.cat(feat2_list, dim=0)
    labels = torch.LongTensor(label_list)
    return feat1, feat2, labels


def _get_label_weight(opts, data):
    """Get number of positive examples for each label."""
    tic = time.time()
    experiments = data["exp_names"].value
    label_mat = np.zeros((experiments.size, 7))
    vid_lengths = np.zeros((experiments.size,))
    for i in range(experiments.size):
        exp_key = experiments[i]
        exp = data["exps"][exp_key]
        for j in range(6):
            # label_counts[j] += exp["org_labels"].value[:, j].sum()
            label_mat[i, j] = exp["labels"].value[:, j].sum()
        # label_counts[-1] +=\
        #     exp["org_labels"].shape[0] - exp["org_labels"].value.sum()
        label_mat[i, -1] =\
            exp["labels"].shape[0] - exp["labels"].value.sum()

        # vid_lengths[i] = exp["hoghof"].shape[0]
        vid_lengths[i] = exp["labels"].shape[0]

    # label_counts = label_mat.sum(axis=0)
    # import pdb; pdb.set_trace()

    label_weight = 1.0 / np.mean(label_mat, axis=0)
    # label_weight = label_mat.sum(axis=0)  / np.mean(label_mat, axis=0)
    # label_weight[-2] = label_weight[-2] * 10
    if opts["flags"].reweight is False:
        label_weight = [5, 5, 5, 5, 5, 5, .01]
    print(time.time() - tic)
    return label_weight


def main(argv):
    opts = _setup_opts(argv)
    paths.setup_output_space(opts)

    if opts["flags"].cuda_device != -1:
        torch.cuda.set_device(opts["flags"].cuda_device)

    with h5py.File(opts["flags"].train_file, "r") as train_data:
        with h5py.File(opts["flags"].test_file, "r") as test_data:
            with h5py.File(opts["flags"].valid_file, "r") as valid_data:
                if DEBUG:
                    train_data = valid_data
                    test_data = valid_data

                sequences_helper.copy_templates(
                    opts, train_data, "train", g_label_names)
                sequences_helper.copy_templates(
                    opts, test_data, "test", g_label_names)
                sequences_helper.copy_templates(
                    opts, valid_data, "valid", g_label_names
                )

                sampler = HantmanVideoFrameSampler(
                    opts["rng"], train_data, opts["flags"].video_dir,
                    opts["flags"].hantman_mini_batch,
                    frames=opts["flags"].frames,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)

                label_weight = _get_label_weight(opts, train_data)
                # import pdb; pdb.set_trace()
                train_eval = HantmanVideoSampler(
                    opts["rng"], train_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)
                test_eval = HantmanVideoSampler(
                    opts["rng"], test_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)
                valid_eval = HantmanVideoSampler(
                    opts["rng"], valid_data, opts["flags"].video_dir,
                    use_pool=True, gpu_id=opts["flags"].cuda_device)

                network, optimizer, criterion = _init_network(
                    opts, label_weight)
                # import pdb; pdb.set_trace()
                _train_network(
                    opts, network, optimizer, criterion,
                    sampler, train_eval, test_eval, valid_eval
                )


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv)
