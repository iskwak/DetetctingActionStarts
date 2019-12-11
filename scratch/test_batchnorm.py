"""Test batch norm layer."""
import theano
# import theano.tensor as T
import numpy
# import layers.losses as losses
import layers.layers as layers
import layers.batch_norm as batch_norm

# some globals.
rng = numpy.random.RandomState(123)
eps = numpy.finfo(numpy.float32).eps

x = theano.tensor.tensor4('x')
is_train = theano.tensor.scalar('is_train')

network = layers.init_network_dict()

chan_out = 4
network['output'] = x

network = batch_norm.spatial_batch_norm_layer(
    rng, network, chan_out, is_train, name=None)

func = theano.function([x, is_train], network['output'],
                       updates=network['updates'])
# moo needs to be batch size by chan out by some image size
# lets do batch size 2, 4 chan out, and 3x3
moo1 = numpy.zeros((1, 4, 3, 3), dtype='float32')
moo2 = numpy.asarray(
    [[[1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]],
     [[2, 2, 2],
      [2, 2, 2],
      [2, 2, 2]],
     [[3, 3, 3],
      [3, 3, 3],
      [3, 3, 3]],
     [[4, 4, 4],
      [4, 4, 4],
      [4, 4, 4]]]).astype('float32').reshape((1, 4, 3, 3))

moo = numpy.concatenate([moo1, moo2])
print func(moo, 1)

import pdb; pdb.set_trace()
