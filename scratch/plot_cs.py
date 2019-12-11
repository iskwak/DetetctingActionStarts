import h5py
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

base_dir = ("/localhome/kwaki/theano-env/QuackNN/quackaction/figs/test_4/"
            "predictions/train/")
# base_dir = ("/localhome/kwaki/temp/moocow/train/")

sub_dirs = ["M173_20150421_v005", "M173_20150424_v057", "M173_20150430_v025",
            "M173_20150430_v065", "M173_20150512_v067"]


def apply_transform(pca, num_components, data):
    """Apply the pca transform using upto num_components."""
    components = pca.components_[:num_components, :]
    transformed = np.dot(data - pca.mean_, components.T)
    return transformed

c1s = []
c2s = []
for sub_dir in sub_dirs:
    exp_dir = os.path.join(base_dir, sub_dir)
    with h5py.File(os.path.join(exp_dir, "states.hdf5"), "r") as h5:
        c1 = h5["c1"].value
        c2 = h5["c2"].value
        c1 = c1[:, 0, :]
        c2 = c2[:, 0, :]

        c1s.append(c1)
        c2s.append(c2)

pca = PCA(n_components=2)
all_c2s = np.concatenate(c2s)

pca.fit(all_c2s)

transformed1 = apply_transform(pca, 2, c2s[0])
transformed2 = apply_transform(pca, 2, c2s[1])
transformed3 = apply_transform(pca, 2, c2s[2])
transformed4 = apply_transform(pca, 2, c2s[3])
transformed5 = apply_transform(pca, 2, c2s[4])

# plt.plot(transformed1[:, 0], transformed2[:, 1], 'ro')
plt.plot(
    transformed1[:, 0], transformed1[:, 1], 'ro',
    transformed2[:, 0], transformed2[:, 1], 'bo',
    transformed3[:, 0], transformed3[:, 1], 'go',
    transformed4[:, 0], transformed4[:, 1], 'ko',
    transformed5[:, 0], transformed5[:, 1], 'co',
)
plt.show()

# plot based on class
all_idx = np.zeros((transformed1.shape[0]), dtype=bool)
lift_idx = all_idx.copy()
hand_idx = all_idx.copy()
grab_idx = all_idx.copy()
sup_idx = all_idx.copy()
other_idx = np.ones((transformed1.shape[0]), dtype=bool)

lift_idx[81:92] = True
other_idx[81:92] = False

hand_idx[104:115] = True
other_idx[104:115] = False
hand_idx[150:160] = True
other_idx[150:160] = False

grab_idx[129:140] = True
other_idx[129:140] = False
grab_idx[168:179] = True
other_idx[168:179] = False

sup_idx[175:186] = True
other_idx[175:186] = False

lift = transformed1[lift_idx, :]
hand = transformed1[hand_idx, :]
grab = transformed1[grab_idx, :]
sup = transformed1[sup_idx, :]
other = transformed1[all_idx, :]

other = np.concatenate([transformed1[:81, :], transformed1[92:, :]])

plt.plot(
    other[:, 0], other[:, 1], 'ko',
    lift[:, 0], lift[:, 1], 'ro',
    hand[:, 0], hand[:, 1], 'go',
    grab[:, 0], grab[:, 1], 'bo',
    sup[:, 0], sup[:, 1], 'co'
)
plt.show()
