import h5py
import numpy
import os


def main():
    base_path = "/nrs/branson/kwaki/data/20180729_base_hantman/exps"
    exps = os.listdir(base_path)
    exps.sort()
    # remove the hoghof folder.
    exps = exps[:-1]

    for exp_name in exps:
        print(exp_name)
        full_name = os.path.join(base_path, exp_name)
        with h5py.File(full_name, 'a') as h5data:
            h5data["hog_side"] = h5py.ExternalLink(
                os.path.join("hoghof", exp_name), "/hog_side"
            )
            h5data["hog_front"] = h5py.ExternalLink(
                os.path.join("hoghof", exp_name), "/hog_front"
            )

            h5data["hof_side"] = h5py.ExternalLink(
                os.path.join("hoghof", exp_name), "/hof_side"
            )
            h5data["hof_front"] = h5py.ExternalLink(
                os.path.join("hoghof", exp_name), "/hof_front"
            )


if __name__ == "__main__":
    main()
