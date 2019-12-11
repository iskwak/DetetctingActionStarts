"""Load and manipulate a sklearn pca object."""
import numpy
import h5py
from sklearn.decomposition import IncrementalPCA
# import shutil
import time
from sklearn.externals import joblib


def apply_transform(pca, num_components, data):
    """Apply the pca transform using upto num_components."""
    components = pca.components_[:num_components, :]
    transformed = numpy.dot(data - pca.mean_, components.T)
    return transformed

# fname = ("/media/drive1/data/hantman_processed/"
#          "relative_19window2/all_feat_sampled.hdf5")
fname = ("/media/drive1/data/hantman_processed/relative_19window2/all.hdf5")
# fname = ("/media/drive1/data/hantman_processed/relative_19window2/"
#          "newsample.hdf5")

ipca_filename = ("/media/drive1/data/hantman_processed/relative_19window2/"
                 "ipca/ipca2.npy")

ipca = joblib.load(ipca_filename)

# load the data
# with h5py.File(fname, "r") as file:
#     data = file["feats"].value

import pdb; pdb.set_trace()
