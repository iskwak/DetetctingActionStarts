"""Look into various losses."""
import h5py
import numpy as np
# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly as py
# import plotly.graph_objs as go
import theano
import theano.tensor as T
import theano.tensor.signal.conv
import helpers.post_processing as post_processing
# import T.signal.conv


h5_filename = "/nrs/branson/kwaki/data/hoghofpos_withorg/data.hdf5"
# h5_filename = "/home/ikwak/research/data/hoghofpos_withorg/data.hdf5"
with h5py.File(h5_filename, "r") as h5_data:
    # the desired experiment is M134_20150316_v001
    exp = "M134_20150316_v001"

    org_labels = h5_data["exps"][exp]["org_labels"].value
    org_labels = org_labels.reshape(
        (org_labels.shape[0], 1, org_labels.shape[1]))
    labels = h5_data["exps"][exp]["labels"].value

    exp2 = h5_data["experiments"].value[2]

    org_labels1 = h5_data["exps"][exp2]["org_labels"].value

    org_labels1 = org_labels1.reshape(
        (org_labels1.shape[0], 1, org_labels1.shape[1]))
    labels1 = h5_data["exps"][exp2]["labels"].value

    # crop...
    num_frames = min(labels1.shape[0], labels.shape[0])
    labels = labels[:num_frames, :, :]
    org_labels = org_labels[:num_frames, :, :]

    labels1 = labels1[:num_frames, :, :]
    org_labels1 = org_labels1[:num_frames, :, :]

    labels = np.concatenate([labels, labels1], axis=1).astype("float32")
    org_labels = np.concatenate([org_labels, org_labels1], axis=1)
    org_labels = org_labels.astype("float32")

# just work with 2 labels.
sub_labels = labels[:, :, :2]
org_sub_labels = org_labels[:, :, :2]
predict = np.zeros(sub_labels.shape, dtype="float32")
np_label0 = np.argwhere(sub_labels[:, 0, 0])
np_label1 = np.argwhere(sub_labels[:, 0, 1])
predict[240:240+len(np_label0), 0, 0] = sub_labels[np_label0[:, 0], 0, 0]
predict[300:300+len(np_label1), 0, 1] = sub_labels[np_label1[:, 0], 0, 1]


def _lstm_step(x_t):
    # dummy scan function.
    # x_t = T.switch(T.lt(x_t, 0.5), 0.0, 1.0)
    return x_t

x_var = T.tensor3("x", dtype=theano.config.floatX)
scan_outputs = [None]
scan_sym, updates = theano.scan(
    _lstm_step,
    sequences=[x_var],
    outputs_info=scan_outputs)

func = theano.function([x_var], scan_sym)

# now apply the hungarian loss.
outputs = func(predict)

val_threshold = 0.7
frame_threshold = 10
y_org = org_sub_labels

COST_FP = 20
COST_FN = 20

num_frames, num_vids, num_classes = outputs.shape
ref_array = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
mask_array = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
num_false_neg = 0
mini_batch_size = 2
for i in range(num_vids):

    for j in range(num_classes):
        processed, max_vals = post_processing.nonmax_suppress(
            outputs[:, i, j], val_threshold)
        processed = processed.reshape((processed.shape[0], 1))
        data = np.zeros((len(processed), 3), dtype="float32")
        data[:, 0] = range(len(processed))
        data[:, 1] = processed[:, 0]
        data[:, 2] = y_org[:, i, j]

        # after suppression, apply hungarian.
        labelled = np.argwhere(y_org[:, i, j] == 1)
        labelled = labelled.flatten().tolist()
        num_labelled = len(labelled)
        # proc_gt = np.zeros((len(labelled), 2), dtype="float32")
        # proc_gt[:, 0] = range(len(labelled))
        # proc_gt[:, 1] = labelled

        # dist_mat = post_processing.create_frame_dists(
        #     processed, max_vals, labelled)
        dist_mat = post_processing.create_frame_dists(
            data, max_vals, labelled)
        rows, cols, dist_mat = post_processing.apply_hungarian(
            dist_mat, frame_threshold)
        rows - cols + dist_mat

        # missed classifications
        false_neg = len(labelled) - len(
            [i for i in range(len(max_vals)) if cols[i] < len(labelled)])
        num_false_neg += false_neg

        false_pos = len(max_vals) - len(
            [i for i in range(len(labelled))
             if np.where(i == cols)[0][0] < len(max_vals)])

        # ref array?
        # ref_array = np.zeros((data.shape[0],), dtype="float32")
        for pos in range(len(max_vals)):
            ref_idx = max_vals[pos]
            if cols[pos] < len(labelled):
                # true positive
                label_idx = labelled[cols[pos]]
                ref_array[ref_idx, i, j] = np.abs(ref_idx - label_idx)
                mask_array[ref_idx, i, j] = 1
            else:
                # false positive
                ref_array[ref_idx, i, j] = COST_FP


def create_cost_func():
    x = T.tensor3("x")
    y = T.tensor3("y")
    # sum_diffs = 10

    cost = x * y
    partial_sum = []
    for i in range(mini_batch_size):
        partial_sum.append(cost[:, i, :].sum(axis=0))

    return theano.function([x, y], [cost] + partial_sum)

moo = create_cost_func()

cow = moo(predict, ref_array)


def step(x_t, ref_t, mask, accum):
    # cost = x_t * ref_t
    # accum = accum + cost
    cost = T.switch(T.isclose(mask, 0),
                    x_t,
                    T.inv(x_t))
    accum = cost + accum
    return cost, accum

mask_var = T.tensor3("mask")
ref_var = T.tensor3("ref")
accum_array = np.zeros((num_vids, num_classes), dtype="float32")
scan_outputs = [None, accum_array]
scan_sym, updates = theano.scan(
    step,
    sequences=[x_var, ref_var, mask_var],
    outputs_info=scan_outputs)

func2 = theano.function([x_var, ref_var, mask_var], scan_sym)
cow = func2(predict, ref_array, mask_array)

temp = T.matrix("t")
temp2 = T.matrix("t2")
beep = theano.function([temp, temp2],
                       T.switch(T.isclose(temp2, 0),
                                temp + 1,
                                temp - 1))
boop = np.zeros((2, 2), dtype="float32")
bloop = np.zeros((2, 2), dtype="float32")
bloop[0, 0] = 1

import pdb; pdb.set_trace()
