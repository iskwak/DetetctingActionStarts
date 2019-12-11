"""Look into various losses."""
import h5py
import numpy as np
# import matplotlib.pyplot as plt
# import plotly.plotly as py
import plotly as py
import plotly.graph_objs as go
import theano
import theano.tensor as T
import theano.tensor.signal.conv
# import T.signal.conv


def mse(label1, label2):
    return np.square(label1 - label2).sum()


def emd(label1, label2):
    return np.sum(np.abs(np.cumsum(label1)-np.cumsum(label2)))


h5_filename = "/nrs/branson/kwaki/data/hoghofpos_withorg/data.hdf5"
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

lift_conv = labels[:, 0, 0]
lift_org = org_labels[:, 0, 0]
print lift_conv.dtype

# ones are at
active_conv = range(233, 252)
active_org = 242

# fake label setups.
predict1 = np.zeros(lift_conv.shape)

predict2 = np.zeros(lift_conv.shape)
predict2[228:247] = lift_conv[active_conv]

predict3 = np.zeros(lift_conv.shape)
predict3[237] = 1

predict4 = np.zeros(lift_conv.shape)
predict4[235:240] = 1.0 / 5.0

predict5 = np.zeros(lift_conv.shape)
predict5[240:243] = 1.0 / 3.0

predict6 = np.zeros(lift_conv.shape)
predict6[241] = 1.0

trace0 = go.Scatter(
    x=np.asarray(range(len(lift_conv)), dtype="float32").T,
    y=lift_org,
    mode="lines+markers",
    name="moocow"
)
trace1 = go.Scatter(
    x=np.asarray(range(len(lift_conv)), dtype="float32").T,
    y=predict3,
    mode="lines+markers",
    name="moocow"
)

data = [trace0, trace1]
py.offline.plot({
    "data": data,
    "layout": go.Layout(title="beep")
},
    filename="figs/moocow/test.html",
    auto_open=False
)

temp = np.asarray([0, 0, 1, 0, 0, 0], dtype="float32")
filt = np.asarray([1, 1, 1], dtype="float32")
print np.convolve(temp, filt, mode="same")

temp = np.asarray([0, 1, 0, 0, 0, 0], dtype="float32")
filt = np.asarray([1, 1, 1], dtype="float32")
temp2 = np.convolve(temp, filt, mode="same")
print temp2
print np.convolve(temp2, filt, mode="same")


def _lstm_step(x_t):
    # dummy scan function.
    x_t = T.switch(T.lt(x_t, 0.5), 0.0, 1.0)
    return x_t

x_var = T.tensor3("x", dtype=theano.config.floatX)
scan_outputs = [None]
scan_sym, updates = theano.scan(
    _lstm_step,
    sequences=[x_var],
    outputs_info=scan_outputs)

func = theano.function([x_var], scan_sym)

conv_data = labels[:, :, :3]
# conv_data = conv_data.reshape((conv_data.shape[0], conv_data.shape[2]))
print func(conv_data).shape
print func(conv_data).dtype

# next try to apply a 1d convolution to the data.
# the convolution operations expects things in [num images] x image height x
# image width. Our "images" are similar to the sequence batch id. So need
# a dim shuffle first.
scan_sym = scan_sym.dimshuffle(1, 0, 2)

func = theano.function([x_var], scan_sym)
print func(conv_data).shape

# conv_filter = np.asarray([[1], [1], [1]], dtype="float32")
conv_filter = np.ones((3, 1), dtype="float32")
conv_filter = theano.shared(value=conv_filter)

conv_sym = T.signal.conv.conv2d(scan_sym, conv_filter, border_mode="full")
conv_sym = T.signal.conv.conv2d(conv_sym, conv_filter, border_mode="full")
func = theano.function([x_var], conv_sym)

test_data1 = np.asarray(
    [[0, .75], [.25, 1], [1, .75], [.25, 0], [0, 0], [0, 0]],
    dtype="float32").reshape(
        (6, 1, 2))
test_data2 = np.asarray(
    [[0, 0], [0, 1], [0, 0], [0, 0], [1, 0], [0, 0]],
    dtype="float32").reshape(
        (6, 1, 2))
test_data = np.concatenate([test_data1, test_data2], axis=1)
print func(test_data)
# print func(conv_data).shape
# print func(conv_data).dtype
