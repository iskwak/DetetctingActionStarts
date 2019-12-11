"""Test dropout layer."""
import theano
# import theano.tensor as T
import numpy
# import layers.losses as losses
import layers.layers as layers

# some globals.
rng = numpy.random.RandomState(123)
eps = numpy.finfo(numpy.float32).eps

x = theano.tensor.matrix('x')
is_train = theano.tensor.scalar('is_train')
layer = layers.create_dropout_layer(x, is_train, prob=0.5, rng=None)

func = theano.function([x, is_train], layer)
moo = numpy.asarray(
    [[1, 2, 3],
     [4, 5, 6]]).astype('float32')
import pdb; pdb.set_trace()
