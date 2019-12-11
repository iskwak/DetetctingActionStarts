"""Convert matlab mouse data into a pickle."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time

# fname = ("/media/drive1/data/hantman_processed/"
#          "relative_19window2/all_feat_sampled.hdf5")
fname = ("/media/drive1/data/hantman_processed/relative_19window2/all.hdf5")
# fname = ("/media/drive1/data/hantman_processed/relative_19window2/"
#          "newsample.hdf5")

with h5py.File(fname, "r") as file:
    data = file["feats"].value

batch_size = 1000

(num_feat, num_dims) = data.shape
num_components = min([num_feat, num_dims])
ipca = IncrementalPCA(n_components=500, batch_size=batch_size)
moo = range(num_feat)

import pdb; pdb.set_trace()
num_batch = int(numpy.ceil(1.0 * num_feat / batch_size))
step = int(numpy.floor(1.0 * num_feat / batch_size)) + 1
tic = time.time()
for i in range(num_batch):
    print "(%d, %d)" % (i, num_batch)
    # print data[i:num_feat:num_batch, :].shape
    startidx = i * batch_size
    endidx = (i + 1) * batch_size
    temp = data[startidx:endidx, :]
    print "\tstarting partial fit..."
    ipca.partial_fit(temp)
    print "\t%f" % (time.time() - tic)
    tic = time.time()
import pdb; pdb.set_trace()
