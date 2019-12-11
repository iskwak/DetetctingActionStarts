import h5py
import os


def process_split(split_num, base_path, feature_path, exp_names):
    # for each Suturing experimnet, create a hdf.
    split_dir = os.path.join(
        base_path, "split_%d" % (split_num + 1), "hoghof_eq")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    # after creating the split dir, create symlinks for the feature
    # files.
    for exp_name in exp_names:
        full_exp_name = os.path.join(
            base_path, exp_name
        )
        # create a symlink into the split dir
        # hard code to use capture2...
        capture_name = "%s_capture2" % exp_name
        full_feature_file = os.path.join(
            feature_path, capture_name
        )
        out_file = os.path.join(split_dir, exp_name)
        if not os.path.exists(out_file):
            os.symlink(full_feature_file, out_file)

        # load up the experiment and add the split.
        with h5py.File(full_exp_name, "a") as h5data:
            split_name = "split_%d" % (split_num + 1)
            # if split_name in h5data.keys():
            #     del h5data[split_name]
            # h5data.create_group(split_name)

            h5data[split_name]["hog_eq"] = h5py.ExternalLink(
                os.path.join(split_name, "hoghof_eq", exp_name),
                "/hog"
            )
            h5data[split_name]["hof_eq"] = h5py.ExternalLink(
                os.path.join(split_name, "hoghof_eq", exp_name),
                "/hof"
            )

            
def main():
    base_path = "/nrs/branson/kwaki/data/20180619_jigsaw_base/exps"
    feature_path = "/nrs/branson/kwaki/data/features/jigsaw_hoghof"
    exp_names = os.listdir(base_path)
    # filter out non Suturing videos
    exp_names = [
        exp_name for exp_name in exp_names if "Suturing" in exp_name
    ]
    exp_names.sort()

    # 8 splits, so create a hoghof for each split.
    for i in range(8):
        # for each pslit, create hoghof links
        process_split(i, base_path, feature_path, exp_names)


if __name__ == "__main__":
    main()
