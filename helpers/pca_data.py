"""Apply PCA on data."""
import numpy as np
import h5py
from sklearn.decomposition import IncrementalPCA
import time


h5_fname = "/media/drive1/data/hantman_processed/test/splitstructure/data.hdf5"

batch_size = 100
ipca = IncrementalPCA(n_components=500, batch_size=batch_size)

with h5py.File(h5_fname, "r") as data:
    keys = list(data["exps"].keys())

    # for pca, sample a handful of videos for each mouse
    all_mice = data["mice"].value
    mice = np.unique(all_mice)
    features = []
    # labels = []
    for mouse in mice:
        # for each mouse, sample a few videos
        exps = [exp for exp in keys if mouse in exp]
        sampled = np.random.choice(exps, 10, replace=False)

        # concatenate the features
        tic = time.time()
        features += [data["exps"][key]["features"].value for key in sampled]
        # labels += [data["exps"][key]["labels"].value for key in sampled]
        features = np.concatenate(features)

        ipca.partial_fit(features)
        toc = time.time() - tic
        print("Finished a batch... %d seconds" % toc)
        import pdb; pdb.set_trace()
