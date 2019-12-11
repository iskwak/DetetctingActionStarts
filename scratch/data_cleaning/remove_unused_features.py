# used features:
# canned_i3d_rgb_64, hoghof, <mouse>_finetune2_i3d_rgb
import h5py
import os


def main():
    base_dir = '/groups/branson/bransonlab/kwaki/data/mice_release/mousereach/hdf5/exps'
    exps = os.listdir(base_dir)
    exps.sort()
    for exp_name in exps:
        exp_name = os.path.join(base_dir, exp_name)
        with h5py.File(exp_name, "a") as h5_data:
            h5_keys = list(h5_data.keys())
            for key in h5_keys:
                if key == "labels" or key == "hoghof":
                    continue
                del h5_data[key]
            print(h5_data.keys())

if __name__ == "__main__":
    main()
