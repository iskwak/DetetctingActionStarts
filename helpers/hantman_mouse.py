"""Helpers to deal with mouse data."""
import numpy
# import theano
from sklearn.externals import joblib
import os
import helpers.paths as paths


def sample_mouse_dates(opts, all_mice, all_dates, mouse):
    """Helper to sample dates for videos for training/testing."""
    # get the mice idx.
    rng = opts["rng"]
    mice_idx = numpy.argwhere(mouse == all_mice)
    mice_dates = numpy.unique(all_dates[mice_idx])
    combined = list(zip(all_mice, all_dates))
    vid_dates = []
    vid_counts = []
    num_vids = 0
    for date in mice_dates:
        vids = [exp for exp in combined
                if (exp[0] == mouse) and (exp[1] == date)]
        vid_dates.append(date)
        vid_counts.append(len(vids))
        num_vids += len(vids)

    # convert to numpy arrays
    vid_counts = numpy.asarray(vid_counts)
    vid_dates = numpy.asarray(vid_dates)

    # rand_idx = rng.permutation(len(vid_dates))
    rand_idx = rng.permutation(len(vid_dates))
    cumsum = numpy.cumsum(vid_counts[rand_idx]) * 1.0 /\
        num_vids

    split_idx = numpy.argwhere(cumsum > .8)[0][0]
    train_dates = vid_dates[rand_idx[:split_idx]]
    train_counts = vid_counts[rand_idx[:split_idx]]
    test_dates = vid_dates[rand_idx[split_idx:]]
    test_counts = vid_counts[rand_idx[split_idx:]]
    print("Training Videos: %d" % sum(train_counts))
    print("Testing Videos: %d" % sum(test_counts))

    return train_dates, test_dates


def setup_full_split(opts, h5_data, mask=None):
    """Setup train/test/val split on all the mice data."""
    # In this paradigm, the network should see no videos of the mice in
    # the valid, and then in the test either.
    all_mice = h5_data["mice"].value
    all_dates = h5_data["date"].value
    if mask is not None:
        all_mice = all_mice[mask]
        all_dates = all_dates[mask]

    unique_mice = numpy.unique(all_mice)
    mouse_vid_counts = numpy.zeros(unique_mice.size)

    # get video counts for each mouse.
    for i in range(len(unique_mice)):
        mouse = unique_mice[i]
        num_vid = (all_mice == mouse).sum()
        mouse_vid_counts[i] = num_vid

    descend_idx = numpy.flip(numpy.argsort(mouse_vid_counts), axis=0)
    sorted_mice = unique_mice[descend_idx]
    # # the order is, train vids, test vids, valid vids.
    # train_mice = sorted_mice[:-2]
    # test_mice = sorted_mice[-2]
    # valid_mice = sorted_mice[-1]
    # new test different mouse setup
    train_mice = sorted_mice[[0, 1, 3]]
    test_mice = sorted_mice[2]
    valid_mice = sorted_mice[-1]

    # create the splits
    train_idx = create_split(all_mice, train_mice)
    test_idx = create_split(all_mice, [test_mice])
    valid_idx = create_split(all_mice, [valid_mice])

    return train_idx, test_idx, valid_idx


def setup_full_split2(opts, h5_data, mask=None, test_mouse="M174"):
    """Setup train/test/val split on all the mice data."""
    # In this paradigm, the test mouse is seperate from all the train mice. The
    # val exps are the last day of mice seen during training.
    all_mice = h5_data["mice"].value
    all_dates = h5_data["date"].value

    if mask is not None:
        all_mice = all_mice[mask]
        all_dates = all_dates[mask]

    unique_mice = numpy.unique(all_mice)
    # train_idx = numpy.zeros(
    #     all_mice.shape, dtype=bool
    # )
    test_idx = numpy.zeros(
        all_mice.shape, dtype=bool
    )
    # valid_idx = numpy.zeros(
    #     all_mice.shape, dtype=bool
    # )

    # hardcode mice selection
    # Using M134, M147, M173 for training. And M174 for testing.
    # Validation will be the last day for each training mouse.
    mouse_names = ["M134", "M147", "M173", "M174"]
    test_mice = [test_mouse]
    train_mice = [
        mouse for mouse in mouse_names if mouse != test_mouse
    ]
    # train_mice = ["M134", "M147", "M173"]
    # test_mice = ["M174"]

    train_idx = []
    valid_idx = []
    for mouse in train_mice:
        curr_mouse = all_mice == mouse
        curr_dates = all_dates[curr_mouse]
        unqiue_dates = numpy.unique(curr_dates)
        val_date = unqiue_dates[-1]
        # loop over all the experiments and add the non last day exps to the
        # training set.
        for i in range(len(all_mice)):
            if mouse == all_mice[i]:
                # train or validation?
                if val_date == all_dates[i]:
                    valid_idx.append(i)
                else:
                    train_idx.append(i)

    test_idx = create_split(all_mice, test_mice)

    # # create the splits
    # import pdb; pdb.set_trace()
    # train_idx = create_split(all_mice, train_mice)
    # test_idx = create_split(all_mice, [test_mice])
    # valid_idx = create_split(all_mice, [valid_mice])

    return train_idx, test_idx, valid_idx


def setup_full_split3(opts, h5_data, mask=None):
    """Setup train/test/val split on all the mice data."""
    # actually split up train/test/val, where 1 day for 1 mouse is the test
    # set.
    all_mice = h5_data["mice"].value
    all_dates = h5_data["date"].value

    # some change in packages between python, numpy, and h5py has chnaged how
    # strings are stored or represented. Convert the all_mice and all_dates
    # bytestrings to unicode ("U13").
    all_mice = all_mice.astype("U13")
    all_dates = all_dates.astype("U13")

    if mask is not None:
        all_mice = all_mice[mask]
        all_dates = all_dates[mask]

    test_mouse = opts["flags"].test_mouse
    test_date = opts["flags"].test_date

    train_idx = []
    valid_idx = []
    test_idx = []

    # In this split, set the first day of the test mouse as validation.
    test_mouse_idx = numpy.argwhere(all_mice == test_mouse)
    mouse_dates = numpy.unique(all_dates[test_mouse_idx])

    valid_mouse = test_mouse
    valid_date = mouse_dates[0]
    # make sure valid and test date aren't the same
    if valid_date == test_date:
        print("test and validationd mouse+date are the same!")
        import pdb; pdb.set_trace()

    for idx in range(len(all_mice)):
        if all_mice[idx] == test_mouse\
                and all_dates[idx] == test_date:
            # this is a test mouse+date, add it to the test list.
            test_idx.append(idx)
        elif all_mice[idx] == valid_mouse\
                and all_dates[idx] == valid_date:
            valid_idx.append(idx)
        else:
            # train or validation mouse/date.
            train_idx.append(idx)

    # test_idx = numpy.argwhere(
    #     numpy.logical_and(all_mice == test_mouse, all_dates == test_date)
    # )
    # valid_idx = numpy.argwhere(
    #     numpy.logical_and(all_mice == valid_mouse, all_dates == valid_date)
    # )
    # train_idx = numpy.argwhere(
    #     numpy.logical_or(
    #         all_mice ~= test_mouse, all_dates ~= test_date
    #     )

    # )
    # import pdb; pdb.set_trace()
    # for mouse in train_mice:
    #     curr_mouse = all_mice == mouse
    #     curr_dates = all_dates[curr_mouse]
    #     unqiue_dates = numpy.unique(curr_dates)
    #     val_date = unqiue_dates[-1]
    #     # loop over all the experiments and add the non last day exps to the
    #     # training set.
    #     for i in range(len(all_mice)):
    #         if mouse == all_mice[i]:
    #             # train or validation?
    #             if val_date == all_dates[i]:
    #                 valid_idx.append(i)
    #             else:
    #                 train_idx.append(i)

    # test_idx = create_split(all_mice, test_mice)

    # # create the splits
    # import pdb; pdb.set_trace()
    # train_idx = create_split(all_mice, train_mice)
    # test_idx = create_split(all_mice, [test_mice])
    # valid_idx = create_split(all_mice, [valid_mice])

    return train_idx, test_idx, valid_idx


def create_split(all_mice, split_mice):
    """Create the split for this set of mice."""
    all_idx = []
    for i in range(len(split_mice)):
        idx = numpy.argwhere(all_mice == split_mice[i]).flatten()
        all_idx.append(idx)

    all_idx = numpy.sort(numpy.concatenate(all_idx)).tolist()
    return all_idx


def setup_train_test_samples(opts, h5_data, mask=None):
    """Setup the sampling for h5_data."""
    # For each mouse, split the training and testing data by day.
    all_mice = h5_data["mice"].value
    all_dates = h5_data["date"].value
    if mask is not None:
        all_mice = all_mice[mask]
        all_dates = all_dates[mask]

    mice = numpy.unique(all_mice)
    train_idx = []
    test_idx = []
    for mouse in mice:
        # get the training and testing dates for this mouse
        train_dates, test_dates =\
            sample_mouse_dates(opts, all_mice, all_dates, mouse)
        # train_dates, test_dates =\
        #     sample_seq_mouse_dates(opts, all_mice, all_dates, mouse)
        # make sure the intersection is null
        if numpy.intersect1d(train_dates, test_dates).size != 0:
            print("train/test dates intersection not empty")
            exit()

        # next actually sample the indicies for the training and testing videos
        # loop over the mice idx
        mice_idx = numpy.argwhere(mouse == all_mice)
        for idx in mice_idx:
            # print all_dates[idx[0]]
            if all_dates[idx[0]] in train_dates:
                train_idx.append(idx[0])
            else:
                test_idx.append(idx[0])

    # get the list of dates
    return train_idx, test_idx


def mask_long_vids(h5_data, exp_list):
    # prune videos over 1500 frames...
    exp_mask = numpy.ones((exp_list.shape[0],), dtype="bool")
    num_labels = h5_data["exps"][exp_list[0]]["labels"].shape[2]
    for i in range(exp_list.shape[0]):
        seq_len = h5_data["exps"][exp_list[i]]["labels"].shape[0]
        if seq_len > 1500:
            exp_mask[i] = False
        else:
            if "mask" not in list(h5_data["exps"][exp_list[i]].keys()):
                mask = numpy.zeros((1, 1500, num_labels))
                mask[:, :seq_len, :] = 1
                h5_data["exps"][exp_list[i]]["mask"] = mask

    return exp_mask


# def setup_givens(h5_data, exp_list):
#     """Create a features matrix to dump onto the GPU RAM."""
#     num_vids = len(exp_list)

#     feat_dims = h5_data["exps"][exp_list[0]]["reduced"].shape[2]
#     # remember... seq length x vid number(batch) x features
#     data = numpy.zeros((1500, num_vids, feat_dims), dtype=theano.config.floatX)
#     labels = numpy.zeros((1500, num_vids, 6), dtype=theano.config.floatX)
#     # in order to "mask" the data without masking it... pad with -5's
#     # -1 is still in the range of the features
#     data = data - 5
#     for i in range(len(exp_list)):
#         feat = h5_data["exps"][exp_list[i]]["reduced"].value
#         seq_len = feat.shape[0]
#         data[:seq_len, i, :] = feat.reshape((feat.shape[0], feat.shape[2]))

#         label = h5_data["exps"][exp_list[i]]["labels"].value
#         labels[:seq_len, i, :] = label.reshape(
#             (label.shape[0], label.shape[2]))
#     return data, labels


# def random_mini_batch(opts, h5_data, idx, offset=0, use_mask=False):
#     """Create a mini batch of data."""
#     if "rng" not in list(opts.keys()):
#         rng = numpy.random.RandomState()
#     else:
#         rng = opts["rng"]

#     if "mini_batch_size" in list(opts.keys()):
#         batch_size = opts["mini_batch_size"]
#     else:
#         batch_size = opts["flags"].hantman_mini_batch
#     feat_dims = h5_data["exps"][idx[0]]["reduced"].shape[2]

#     sample_idx = rng.choice(len(idx), replace=False, size=batch_size)
#     # sample_idx = sample_idx + offset

#     if opts["flags"].hantman_givens:
#         return [sample_idx + offset]
#     # for each of the positive indicies, pick a random window around the
#     # index.
#     all_feats = numpy.zeros(
#         (1500, batch_size, feat_dims), dtype=theano.config.floatX)
#     all_feats = all_feats - 5
#     all_labels = numpy.zeros((1500, batch_size, 6), dtype=theano.config.floatX)
#     if use_mask is True:
#         all_masks = numpy.zeros((batch_size, 1500, 6), dtype=theano.config.floatX)

#     for i in range(batch_size):
#         vid_idx = idx[sample_idx[i]]
#         # vid_idx = idx[i]
#         feat = h5_data["exps"][vid_idx]["reduced"].value

#         seq_len = feat.shape[0]
#         all_feats[:seq_len, i, :] =\
#             feat.reshape((feat.shape[0], feat.shape[2]))

#         label = h5_data["exps"][vid_idx]["labels"].value
#         all_labels[:seq_len, i, :] =\
#             label.reshape((label.shape[0], label.shape[2]))

#         if use_mask is True:
#             all_masks[i, :, :] = h5_data["exps"][vid_idx]["mask"].value

#     if use_mask is True:
#         func_inputs = [all_feats, all_labels, all_masks]
#     else:
#         func_inputs = [all_feats, all_labels]

#     return func_inputs, sample_idx + offset


def save_hantman_state(opts, network, train_vids, test_vids, out_name,
                       train_cost=None, test_cost=None):
    """Save the state of the network and training."""
    # first get the parameters of the network.
    params_dict = dict()
    for param in network["params"]:
        params_dict[param.name] = param.get_value()
    grads_dict = dict()
    for grad in network["grad_vals"]:
        grads_dict[grad.name] = grad.get_value()

    optim_dict = dict()
    for optim in network["optim_state"]:
        optim_dict[optim.name] = optim.get_value()

    # next save the train/test split.
    save_dict = {
        "params": params_dict,
        "grads": grads_dict,
        "optim_state": optim_dict,
        "iter": network["lr_update"]["params"][0].get_value(),
        "train_vids": train_vids,
        "test_vids": test_vids,
        "rng": opts["rng"].get_state(),
        "train_cost": train_cost,
        "test_cost": test_cost
    }

    save_dict["rng"] = opts["rng"].get_state()
    joblib.dump(save_dict, out_name)

    return


def load_hantman_state(opts, network, filename):
    """Load the state of the network and training."""
    return


def log_info(opts, train_vids, test_vids):
    """Log some other settings for the training setup."""
    out_dir = os.path.join(
        opts["flags"].out_dir, "info")
    paths.create_dir(out_dir)

    train_txt = os.path.join(out_dir, "train_vids.txt")
    with open(train_txt, "w") as f:
        for train_vid in train_vids:
            f.write("%s\n" % train_vid)

    test_txt = os.path.join(out_dir, "test_vids.txt")
    with open(test_txt, "w") as f:
        for test_vid in test_vids:
            f.write("%s\n" % test_vid)

    info_txt = os.path.join(out_dir, "info.txt")
    with open(info_txt, "w") as f:
        f.write("Num train: %d\n" % len(train_vids))
        f.write("Num test: %d\n" % len(test_vids))
        f.write("Iters per epoch: %d\n" % opts["flags"].iter_per_epoch)
        f.write("Update iterations: %d\n" % opts["flags"].update_iterations)
        f.write("Save iterations: %d\n" % opts["flags"].save_iterations)
