import h5py
import os

dataname = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/train.hdf5"

with h5py.File(dataname, "r") as h5data:
    debug_name = "/groups/branson/bransonlab/kwaki/data/thumos14/h5data/debug.hdf5"
    with h5py.File(debug_name, "w") as debugdata:
        debugdata["label_names"] = h5data["label_names"][()]
        debugdata["exp_names"] = h5data["exp_names"][:10]

        exp_group = debugdata.create_group("exps")
        for i in range(10):
            exp_name = debugdata["exp_names"][i]
            exp_group[exp_name] = h5py.ExternalLink(
                os.path.join("exps", exp_name),
                "/"
            )

