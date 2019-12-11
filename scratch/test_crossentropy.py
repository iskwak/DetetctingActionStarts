"""Test the categorical cross entropy."""
# try to get cross entropy working without a scan...
import theano
import theano.tensor as T
import numpy as np
import layers.losses as losses

# predict_sym = T.tensor3("predict")
# label_sym = T.tensor3("predict3")
# label_idx_sym = T.

predict2d_sym = T.matrix("predict2d")
labelhot_sym = T.ivector("labelhot_sym")

cross_ent = T.nnet.nnet.categorical_crossentropy(
    predict2d_sym, labelhot_sym)

func = theano.function([predict2d_sym, labelhot_sym], cross_ent)

x = np.zeros((3, 4), dtype=theano.config.floatX)
x = x + np.finfo(np.float32).eps
x[0, 2] = 1
x[1, 0] = 1
x[2, 3] = 1

y = np.asarray(
    [[0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]],
    dtype=theano.config.floatX
)

yhot = np.asarray(
    [1, 0, 3],
    dtype="int32"
)

z = func(x, yhot)

test = theano.shared(
    np.asarray(
        [[[1, 2, 3], [12, 11, 10]],
         [[4, 5, 6], [9, 8, 7]],
         [[7, 8, 9], [6, 5, 4]],
         [[10, 11, 12], [3, 2, 1]]],
    ).astype("float32"))

# one hot, needs to be represented as an array?
tensor_one_hot = theano.shared(
    np.asarray(
        [[0, 1, 2],
         [1, 1, 1],
         [2, 1, 0]],
    ).astype("int32")
)

test_sym = T.tensor3("test")
one_hot = T.imatrix("one_hot")

func = theano.function(
    [test_sym, one_hot],
    T.log(-test_sym[one_hot[:, 0],
                    one_hot[:, 1],
                    one_hot[:, 2]]))

func1 = theano.function(
    [test_sym, one_hot],
    losses.sparse_tensor_categorical_cross_entropy(test_sym, one_hot)
)

import pdb; pdb.set_trace()