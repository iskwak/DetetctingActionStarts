import h5py
import os
import numpy
import time
import torch


def init_data_dict(data_dict, h5data, data_keys, max_seq_len):
    # loop over the data keys, and figure out the array sizes.
    num_exp = len(h5data["exp_names"])
    temp_exp = h5data["exps"][h5data["exp_names"][0]]

    # create the labels entry first
    # data_dict = {}
    # data_dict["labels"] = numpy.zeros(
    #     (max_seq_len, num_exp, temp_exp["labels"].shape[1]), dtype=nump.float32
    # )

    # create a video length entry
    data_dict["seq_len"] = numpy.zeros(
        (num_exp,), dtype=numpy.int16
    )

    for key in data_keys:
        feat_dim = temp_exp[key].shape[1]
        data_dict[key] = numpy.zeros(
            (max_seq_len, num_exp, feat_dim), dtype=numpy.float32
        )

    return data_dict


def load_data(h5_fname, data_keys, max_seq_len):
    data_dict = {}
    fid = open("/nrs/branson/kwaki/test/memtest/full_load.csv", "w")

    with h5py.File(h5_fname, "r") as h5data:
        print("init")
        init_data_dict(data_dict, h5data, data_keys, max_seq_len)

        print("loading")
        exps = h5data["exp_names"]
        total_tic = time.time()
        for i in range(len(exps)):
            tic = time.time()
            exp = exps[i]
            print(exp)
            num_frames = h5data["exps"][exp]["labels"].shape[0]
            max_idx = numpy.min(
                [max_seq_len, num_frames]
            )

            data_dict["seq_len"][i] = max_idx
            for key in data_keys:
                data_dict[key][:max_idx, i, :] = \
                    h5data["exps"][exp][key][:max_idx, :]

            # print("\t%f" % (time.time() - tic))
            fid.write("load,%f\n" % (time.time() - tic))
            fid.flush()
        print("total time: %f" % (time.time() - total_tic))
        # for exp in exps:

    fid.close()
    return data_dict

def main():
    torch.cuda.set_device(0)
    data_keys = [
        "rgb_i3d_view1_fc_64", "rgb_i3d_view2_fc_64",
        "flow_i3d_view1_fc", "flow_i3d_view2_fc",
        "labels"
    ]
    # data_keys = ["rgb_i3d_view1_fc_64", "labels"]
    max_seq_len = 1500

    print("load training")
    h5_name = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_train.hdf5'
    train_data = load_data(h5_name, data_keys, max_seq_len)

    # print("load testing")
    # h5_name = '/nrs/branson/kwaki/data/20180729_base_hantman/hantman_split_half_M134_test.hdf5'
    # test_data = load_data(h5_name, data_keys, max_seq_len)

    import pdb; pdb.set_trace()
    print("hi")
    # for i in range(100):



if __name__ == "__main__":
    main()
