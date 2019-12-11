from __future__ import print_function,division
import h5py
import numpy
import os

# exp_path = "/media/drive1/data/hantman_processed/onemouse_flow2/exps"
exp_path = "/media/drive1/data/hantman_processed/20170827_vgg/exps"

exp_names = os.listdir(exp_path)
exp_names.sort()
exp_name = [
    [exp_name for exp_name in exp_names if "M135" in exp_name]
]

for i in range(len(exp_names)):
    exp_name = exp_names[i]
    print(exp_name)

    full_exp_path = os.path.join(exp_path, exp_name)
    with h5py.File(full_exp_path, "a") as exp_data:
        reshape_keys = [
            "img_side_norm", "img_front_norm",
            "paw_side_norm", "paw_front_norm"
        ]

        all_keys = exp_data.keys()
        for key in all_keys:
            # import pdb; pdb.set_trace()
            if "img_side_norm" in exp_data:
                if key in reshape_keys:
                    feats = exp_data[key].value
                    feats = feats.reshape(
                        (feats.shape[0], 1, feats.shape[1]))
                    del exp_data[key]
                    exp_data[key] = feats

    # old_file = os.path.join(old_path, exp_name)
    # new_file = os.path.join(new_path, exp_name)
    # with h5py.File(old_file, "r") as old_data:
    #     with h5py.File(new_file, "w") as new_data:
    #         reshape_keys = [
    #             "side_flow", "front_flow", "img_side_norm", "img_front_norm"
    #         ]

    #         all_keys = old_data.keys()
    #         for key in all_keys:
    #             import pdb; pdb.set_trace()
    #             if "side_flow" in old_data:
    #                 if key in reshape_keys:
    #                     feats = old_data[key].value
    #                     feats = feats.reshape(
    #                         (feats.shape[0], 1, feats.shape[1]))
    #                     new_data[key] = feats
    #                 else:
    #                     new_data[key] = old_data[key].value
