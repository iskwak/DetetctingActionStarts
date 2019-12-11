"""Convert matlab mouse data into a pickle."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from helpers.RunningStats import RunningStats
from sklearn.externals import joblib

fname = "/media/drive1/data/hantman_processed/20170718/data.hdf5"
out_name = "/media/drive1/data/hantman_processed/20170718/stats/stats.pkl"

# after getting the variance, use the max variance to reduce the values of the
# pca'ed data.
with h5py.File(fname, "a") as h5_data:
    exps = list(h5_data["exps"].keys())
    exps.sort()

    count = 0
    # for exp in exps:
    # for i in range(len(sampled)):
    for i in range(len(exps)):
        exp = exps[i]
        print("%d of %d: %s" % (i, len(exps), exp))
        tic = time.time()
        hoghof = h5_data["exps"][exp]["hoghof_norm"]
        pos_features = h5_data["exps"][exp]["pos_norm"]

        temp = numpy.concatenate(
            [hoghof, pos_features], axis=1)
        temp = temp.reshape((temp.shape[0], 1, temp.shape[1]))
        h5_data["exps"][exp]["reduced"] = temp
        print("\t%f" % (time.time() - tic))

import pdb; pdb.set_trace()

