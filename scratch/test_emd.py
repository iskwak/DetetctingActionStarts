"""Test emd distance on LSTMs."""
import theano
import numpy as np
import theano.tensor as T

num_out = 3
num_batch = 4
seq_len = 5


def _scan_step(y_t, yhat_t, y_cumsum, yhat_cumsum):
    """Scan step."""
    idx2 = T.isclose(yhat_t, -100)
    # sub tensor assignment is a bit strange in theano. It's easier to make
    # weight multiplier and modify the whole thing.
    weight = theano.shared(value=np.ones(
        (num_batch, num_out),
        dtype=theano.config.floatX), name="weight_idx")

    # set the weight of things that are "masked" to 0
    weight = weight - idx2

    # yhat_t = yhat_t * weight
    # y_t = y_t * weight

    y_cumsum = y_t + y_cumsum
    yhat_cumsum = yhat_t + yhat_cumsum

    cost = T.abs_(y_cumsum - yhat_cumsum)
    cost = cost * weight
    return cost, y_cumsum, yhat_cumsum


y_t = T.tensor3("y_t")
yhat_t = T.tensor3("yhat_t")
init_y_cumsum = theano.shared(
    np.zeros((num_batch, num_out),
             dtype=theano.config.floatX))
init_yhat_cumsum = theano.shared(
    np.zeros((num_batch, num_out),
             dtype=theano.config.floatX))

scan_outputs = [None, init_y_cumsum, init_yhat_cumsum]

outputs, updates = theano.scan(
    _scan_step,
    sequences=[y_t, yhat_t],
    outputs_info=scan_outputs
)

output = outputs[0] - outputs[1]
output = outputs

func = theano.function([y_t, yhat_t], output)

y = np.zeros((seq_len, num_batch, num_out), dtype="float32")
yhat = np.zeros((seq_len, num_batch, num_out), dtype="float32")
for i in range(seq_len):
    for j in range(num_batch):
        for k in range(num_out):
            y[i, j, k] = i + j + k
            yhat[i, j, k] = i - j - k
yhat[4, 2, :] = -100
yhat[3, 3, :] = -100
yhat[4, 3, :] = -100

moo = theano.function([y_t],  T.isclose(y_t, 0))

import pdb; pdb.set_trace()
# print func(y, yhat)
