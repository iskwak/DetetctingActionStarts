"""Test some scan masking code."""
import numpy as np
# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly as py
# import plotly.graph_objs as go
import theano
import theano.tensor as T
import theano.tensor.signal.conv
# import T.signal.conv


def mse(label1, label2):
    return np.square(label1 - label2).sum()


def emd(label1, label2):
    return np.sum(np.abs(np.cumsum(label1)-np.cumsum(label2)))


def _lstm_step(x_t):
    # dummy scan function.
    x_t = T.switch(T.lt(x_t, 0.5), 0.0, 1.0)
    return x_t

x_var = T.tensor3("x", dtype=theano.config.floatX)
mask_var = T.tensor3("mask", dtype=theano.config.floatX)
scan_outputs = [None]
scan_sym, updates = theano.scan(
    _lstm_step,
    sequences=[x_var],
    outputs_info=scan_outputs)

# next try to apply a 1d convolution to the data.
# the convolution operations expects things in [num images] x image height x
# image width. Our "images" are similar to the sequence batch id. So need
# a dim shuffle first.
scan_sym = scan_sym.dimshuffle(1, 0, 2)

func = theano.function([x_var], scan_sym)

# conv_filter = np.asarray([[1], [1], [1]], dtype="float32")
conv_filter_data = np.ones((5, 1), dtype="float32")
filter_rows = conv_filter_data.shape[0] / 2
conv_filter = theano.shared(value=conv_filter_data)

# conv_sym = scan_sym
conv_sym = T.signal.conv.conv2d(scan_sym, conv_filter, border_mode="full")
conv_sym = conv_sym[:, filter_rows:conv_sym.shape[1] - 2, :]
conv_sym = mask_var * conv_sym
conv_sym = T.signal.conv.conv2d(conv_sym, conv_filter, border_mode="full")
conv_sym = conv_sym[:, 2:conv_sym.shape[1] - 2, :]
conv_sym = mask_var * conv_sym
conv_sym = conv_sym.dimshuffle(1, 0, 2)
conv_sym = T.sum(conv_sym, axis=(0, 2))
func = theano.function([x_var, mask_var], conv_sym)

test_data1 = np.asarray(
    [[0, .75, 0], [.25, 1, 0], [1, .75, 0], [.25, 0, 0], [0, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    dtype="float32").reshape(
        (10, 1, 3))
mask1 = np.zeros((10, 1, 3), dtype="float32")
mask1[:8, :, :] = 1

test_data2 = np.asarray(
    [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0],
     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    dtype="float32").reshape(
        (10, 1, 3))
mask2 = np.zeros((10, 1, 3), dtype="float32")
mask2[:5, :, :] = 1
mask = np.concatenate([mask1, mask2], axis=1)
mask = np.transpose(mask, axes=(1, 0, 2))
print mask.shape
# print mask1
# print mask2

test_data = np.concatenate([test_data1, test_data2], axis=1)
# print test_data[:, 0, :]
# print test_data[:, 1, :]
import pdb; pdb.set_trace()
print func(test_data, mask).shape
# print func(conv_data).shape
# print func(conv_data).dtype

temp = test_data > .5
temp1 = test_data
temp1[temp] = 1
temp1[np.logical_not(temp)] = 0
out1 = np.asarray([
    np.convolve(temp1[:, 0, 0].reshape(10,),
                conv_filter_data[:, 0], mode="same"),
    np.convolve(temp1[:, 0, 1].reshape(10,),
                conv_filter_data[:, 0], mode="same")]).transpose()
# print out1
out1 = np.asarray([
    np.convolve(temp1[:, 1, 0].reshape(10,),
                conv_filter_data[:, 0], mode="same"),
    np.convolve(temp1[:, 1, 1].reshape(10,),
                conv_filter_data[:, 0], mode="same")]).transpose()
# print "moo"
# print out1
# print out1 * mask2.reshape(10, 2)
# print out1
out1 = out1 * mask2.reshape(10, 2)
out1 = np.asarray([
    np.convolve(out1[:, 0].reshape(10,),
                conv_filter_data[:, 0], mode="same"),
    np.convolve(out1[:, 1].reshape(10,),
                conv_filter_data[:, 0], mode="same")]).transpose()
print "cow"
# print out1
print out1 * mask2.reshape(10, 2)

cow = temp1[:5, 1, :]
# print cow
cow = cow.reshape(5, 2)

out1 = np.asarray([
    np.convolve(cow[:, 0].reshape(5,),
                conv_filter_data[:, 0], mode="same"),
    np.convolve(cow[:, 1].reshape(5,),
                conv_filter_data[:, 0], mode="same")]).transpose()

out1 = np.asarray([
    np.convolve(out1[:, 0].reshape(5,),
                conv_filter_data[:, 0], mode="same"),
    np.convolve(out1[:, 1].reshape(5,),
                conv_filter_data[:, 0], mode="same")]).transpose()
print "beep"
print out1
