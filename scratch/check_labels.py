"""Looop over the HDF files and figure out label counts."""
import h5py
import numpy as np
import helpers.hantman_mouse as hantman_mouse
import time

filename = "/nrs/branson/kwaki/data/hoghofpos_withorg/data.hdf5"

opts = {}
opts["rng"] = np.random.RandomState(123)

label_counts = {
    "lift": [],
    "hand": [],
    "grab": [],
    "supp": [],
    "mout": [],
    "chew": [],
    "noth": [],
    "frames": [],
}

with h5py.File(filename, "r") as h5_data:
    tic = time.time()
    exp_list = h5_data["experiments"].value
    exp_mask = hantman_mouse.mask_long_vids(h5_data, exp_list)
    train_vids, test_vids = hantman_mouse.setup_train_test_samples(
        opts, h5_data, exp_mask)

    # now loop over all the videos, ignore training and testing?
    masked = exp_list[exp_mask]

    for exp in masked:
        org_labels = h5_data["exps"][exp]["org_labels"]
        label_counts["frames"].append(org_labels.shape[0])

        lift = np.argwhere(org_labels[:, 0])
        hand = np.argwhere(org_labels[:, 1])
        grab = np.argwhere(org_labels[:, 2])
        supp = np.argwhere(org_labels[:, 3])
        mout = np.argwhere(org_labels[:, 4])
        chew = np.argwhere(org_labels[:, 5])

        label_counts["lift"].append(lift.size)
        label_counts["hand"].append(hand.size)
        label_counts["grab"].append(grab.size)
        label_counts["supp"].append(supp.size)
        label_counts["mout"].append(mout.size)
        label_counts["chew"].append(chew.size)
        label_counts["noth"].append(
            org_labels.shape[0] -
            (lift.size + hand.size + grab.size + supp.size + mout.size +
             chew.size))

        # if lift.size > 1 or hand.size > 1 or grab.size > 1 or\
        #         supp.size > 1 or mout.size > 1 or chew.size > 1:
        #     import pdb; pdb.set_trace()
        #     print "moo"
    print "Time: %d seconds" % (time.time() - tic)

    print "hi"

lift_count = 1.0 * sum(label_counts["lift"]) / sum(label_counts["frames"])
hand_count = 1.0 * sum(label_counts["hand"]) / sum(label_counts["frames"])
grab_count = 1.0 * sum(label_counts["grab"]) / sum(label_counts["frames"])
supp_count = 1.0 * sum(label_counts["supp"]) / sum(label_counts["frames"])
mout_count = 1.0 * sum(label_counts["mout"]) / sum(label_counts["frames"])
chew_count = 1.0 * sum(label_counts["chew"]) / sum(label_counts["frames"])

print 1.0 / sum(label_counts["lift"])
print 1.0 / sum(label_counts["hand"])
print 1.0 / sum(label_counts["grab"])
print 1.0 / sum(label_counts["supp"])
print 1.0 / sum(label_counts["mout"])
print 1.0 / sum(label_counts["chew"])
print 1.0 / sum(label_counts["noth"])
import pdb; pdb.set_trace()

# calculate weighting
totals = lift_count + hand_count + grab_count + supp_count + mout_count\
    + chew_count
noth_count = 1 - totals
print noth_count

print "moo"
print lift_count
print hand_count
print grab_count
print supp_count
print mout_count
print chew_count
print noth_count
#
print "####################################################################"
zipped = zip(label_counts["lift"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["hand"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["grab"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["supp"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["mout"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["chew"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)

zipped = zip(label_counts["noth"], label_counts["frames"])
print 1 - sum([1.0 * a[0] / a[1] for a in zipped]) / len(zipped)
