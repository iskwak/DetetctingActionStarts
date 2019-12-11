"""Helper to create more permanent train/test splits."""
import sys
import os
import numpy
import h5py
import gflags
import helpers.arg_parsing as arg_parsing
import helpers.paths as paths
import helpers.git_helper as git_helper
import helpers.hantman_mouse as hantman_mouse

gflags.ADOPT_module_key_flags(arg_parsing)
gflags.DEFINE_string("data", None, "Data to split into train/test")
gflags.DEFINE_string("name", None, "Name of the split to use")
gflags.DEFINE_boolean("prune", True, "Prune weird videos...")
gflags.DEFINE_boolean("one_day", False, "One day of videos")
gflags.DEFINE_boolean("one_mouse", False, "One mouse of videos")
gflags.DEFINE_string(
    "test_mouse", "M173",
    "Test mouse for cross validation splits.")
gflags.DEFINE_string(
    "test_date", "",
    "Test date for cross validation splits."
)
gflags.DEFINE_integer(
    "split_type", None,
    "Train/Test/Val split type."
)

gflags.MarkFlagAsRequired("data")
gflags.MarkFlagAsRequired("name")
gflags.MarkFlagAsRequired("split_type")


def create_train_test(opts):
    """Create the training and testing splits."""
    # first setup the output space. The output space will be in the same folder
    # as the original data.hdf file, but with different names and a seperate
    # sub folder for the setup information.
    base_out = os.path.dirname(opts["flags"].data)
    log_output_path = os.path.join(base_out, opts["flags"].name)
    # out_data_name = os.path.join(base_out, opts["flags"].name + ".hdf5")
    base_out_name = os.path.join(base_out, opts["flags"].name)
    exp_path = os.path.join(base_out, "exps")

    paths.create_dir(log_output_path)

    # add the initial logging information to the output path.
    git_helper.log_git_status(
        os.path.join(log_output_path, "00_git_status.txt"))
    paths.save_command2(log_output_path, opts["argv"])

    # now to do the actual splitting.
    # first open the base data.hdf
    with h5py.File(opts["flags"].data, "r") as org_data:
        exp_list = org_data["exp_names"].value

        exp_mask = numpy.ones((exp_list.shape[0],), dtype="bool")

        if opts["flags"].split_type == 2:
            train_idx, test_idx, valid_idx = hantman_mouse.setup_full_split2(
                opts, org_data, mask=exp_mask, test_mouse=opts["flags"].test_mouse)
        else:
            # else split type 3... no split type 1
            train_idx, test_idx, valid_idx = hantman_mouse.setup_full_split3(
                opts, org_data, mask=exp_mask
            )

        split_name = base_out_name + "_train.hdf5"
        save_experiments(opts, org_data, exp_path, train_idx, split_name,
                         mask=exp_mask)
        split_name = base_out_name + "_test.hdf5"
        save_experiments(opts, org_data, exp_path, test_idx, split_name,
                         mask=exp_mask)
        split_name = base_out_name + "_valid.hdf5"
        save_experiments(opts, org_data, exp_path, valid_idx, split_name,
                         mask=exp_mask)

    return


def prune_mice_dates(opts, org_data, mask=None):
    """Create experiments mask for one day or one mouse or both."""
    # exp_list = org_data["experiments"].value
    mice_list = org_data["mice"].value
    date_list = org_data["date"].value
    unique_dates = numpy.unique(date_list[mask])
    unique_mice = numpy.unique(mice_list[mask])

    mouse_counts = numpy.zeros(unique_mice.shape)
    date_counts = numpy.zeros(unique_dates.shape)
    mouse_date_counts = numpy.zeros(
        (unique_mice.shape[0], unique_dates.shape[0])
    )
    for i in range(len(unique_mice)):
        for j in range(len(unique_dates)):
            mouse = unique_mice[i]
            date = unique_dates[j]

            num_mouse_day = numpy.sum(
                (date == date_list) *
                (mouse == mice_list) *
                mask
            )
            # DEBUG = numpy.sum(
            #     (date == date_list) *
            #     (mouse == mice_list)
            # )
            # if DEBUG != num_mouse_day:
            #     import pdb; pdb.set_trace()
            mouse_counts[i] += num_mouse_day
            date_counts[j] += num_mouse_day
            mouse_date_counts[i, j] = num_mouse_day

    if opts["flags"].one_day and opts["flags"].one_mouse:
        # numpy.argmax flattens the array... don't feel like figuring out which
        # axis gets flattened first.
        max_val = numpy.max(mouse_date_counts)
        # numpy.arghwere returns a list of maxs. just take the first.
        max_idxs = numpy.argwhere(mouse_date_counts == max_val)
        max_mouse = unique_mice[max_idxs[0][0]]
        max_date = unique_dates[max_idxs[0][1]]

        new_mask = (mice_list == max_mouse) * (date_list == max_date) * mask

    elif opts["flags"].one_day and not opts["flags"].one_mouse:
        max_date = unique_dates[numpy.argmax(date_counts)]
        new_mask = (date_list == max_date) * mask

    elif opts["flags"].one_mouse and not opts["flags"].one_day:
        max_mouse = unique_mice[numpy.argmax(mouse_counts)]
        new_mask = (mice_list == max_mouse) * mask
    else:
        new_mask = mask

    import pdb; pdb.set_trace()
    return new_mask


def save_experiments(opts, org_data, exp_path, split_idx, split_name,
                     mask=None):
    """Save the data subset to file."""
    exp_list = org_data["exp_names"].value
    mice_list = org_data["mice"].value
    date_list = org_data["date"].value
    label_names = org_data["label_names"].value
    if mask is not None:
        exp_list = exp_list[mask]
        mice_list = mice_list[mask]
        date_list = date_list[mask]
    exp_list = exp_list[split_idx]
    mice_list = mice_list[split_idx]
    date_list = date_list[split_idx]

    with h5py.File(split_name, "w") as split_data:
        split_data.create_group("exps")
        split_data["exp_names"] = exp_list
        split_data["mice"] = mice_list
        split_data["date"] = date_list
        split_data["label_names"] = label_names

        for exp_name in exp_list:
            exp_name = exp_name.decode("utf-8")
            full_exp_path = os.path.join(exp_path, exp_name)
            split_data["exps"][exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name), "/"
            )

    return date_list


def _setup_opts(argv):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    FLAGS = gflags.FLAGS

    opts = arg_parsing.setup_opts(argv, FLAGS)
    return opts


if __name__ == "__main__":
    opts = _setup_opts(sys.argv)
    create_train_test(opts)
