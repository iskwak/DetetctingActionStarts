"""Test windowed emd implementation."""
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal.conv


def create_convolution():
    x = T.tensor3("x")
    mask = T.tensor3("mask")
    x_shuf = x.dimshuffle(1, 0, 2)

    filt_data = np.ones((3, 1), dtype="float32")
    filt_rows = filt_data.shape[0] / 2
    filt_sym = theano.shared(value=filt_data)

    conv_sym = T.signal.conv.conv2d(x_shuf, filt_sym, border_mode="full")
    conv_sym = conv_sym[:, filt_rows:conv_sym.shape[1] - filt_rows, :]
    # conv_sym = conv_sym * mask

    conv_sym = T.signal.conv.conv2d(
        conv_sym, filt_sym, border_mode="full"
    )
    conv_sym = conv_sym[:, filt_rows:conv_sym.shape[1] - filt_rows, :]
    # conv_sym = conv_sym * mask

    conv_sym = conv_sym.dimshuffle(1, 0, 2)
    # return [x, mask], conv_sym, T.sum(conv_sym, axis=(0, 2))
    return [x], conv_sym, T.sum(conv_sym, axis=(0, 2))

x1 = np.asarray(
    [[1, 0, 1],
     [1, 1, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
).astype("float32")
x1 = x1.reshape((5, 1, 3)).astype("float32")

x2 = np.asarray(
    [[0, 0, 0],
     [0, 1, 1],
     [1, 1, 0],
     [0, 1, 0],
     [0, 0, 0]]
).astype("float32")
x2 = x2.reshape((5, 1, 3)).astype("float32")

data = np.concatenate([x1, x2], axis=1)
mask = np.ones(data.shape, dtype="float32")

# func_input, out1, out2 = create_convolution()
# func = theano.function(func_input, [out1, out2])

x = T.tensor3("x")
filt = np.asarray([[1], [1], [1], [0], [0]]).astype("float32")
filt_sym = theano.shared(filt)
conv = T.signal.conv.conv2d(
    x.dimshuffle(1, 0, 2), filt_sym, border_mode="valid")
moo = theano.function([x], conv)

import pdb; pdb.set_trace()