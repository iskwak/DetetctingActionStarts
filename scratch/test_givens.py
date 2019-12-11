"""Test theano.function givens."""
import theano
import theano.tensor as T
import numpy
import helpers.load_cifar as load_cifar
import layers.vgg_module as vgg_module
import layers.layers as layers
import layers.losses as losses
import time

# some globals.
rng = numpy.random.RandomState(123)
eps = numpy.finfo(numpy.float32).eps

# load data
x_train, labels_train, x_test, labels_test = load_cifar.cifar10(
    dtype=theano.config.floatX, grayscale=False, data_dir='data')

opts = {
    'rng': rng,
    'img_dims': 3,
    'weight_decay': 1e-4,
    'decay_step': 0,
    'learning_rate': 1e-3,
    'use_bn': True,
    'use_shared': False
}
# create base (forward) network
rng = opts['rng']

is_train = T.scalar('is_train')
x = T.tensor4('x')

network = layers.init_network_dict()
network['output'] = x
network['variables'] = [x, is_train]
network['predict_variables'] = [x, is_train]
# conv1 3x3/64
network = vgg_module.vgg_module(
    rng, network, opts['img_dims'], 64, is_train, name='conv1')
# add dropout
network['output'] = layers.create_dropout_layer(
    network['output'], is_train, prob=0.30)

network = vgg_module.vgg_module(
    rng, network, 64, 64, is_train, name='conv2')

# 64 x 16 x 16 after pooling
network['output'] = T.signal.pool.pool_2d(
    network['output'], (2, 2), mode='max', ignore_border=True)

# batch x 64 x 16 x 16
network['output'] = network['output'].flatten(2)
w = layers.create_weight_matrix(rng, 16384, 10, name='fc_1_w')
b_values = numpy.zeros((10,), dtype=theano.config.floatX)
b = theano.shared(value=b_values, name="fc_1_b", borrow=True)
network['output'] = T.dot(network['output'], w) + b

yhat = T.ivector('yhat')
network['variables'].append(yhat)

network = losses.softmax(network, yhat)

network['cost'] = network['py_cost'].mean()

print "Creating prediction network..."
tic = time.time()
network['prediction'] = theano.function(
    network['predict_variables'], network['y_pred'])
toc = time.time()
print "... took %d seconds" % (toc - tic)

network = layers.weight_decay(
    network, opts['weight_decay'])

# func = theano.function(network['predict_variables'], network['output'])
cow = numpy.zeros((2, 3, 32, 32), dtype='float32')

index = T.ivector('index')
moo = x * 1
cow_shared = theano.shared(x_train)
func = theano.function(
    [index], moo,
    givens={
        x: cow_shared[index, :, :, :]
    })

idx = numpy.asarray([0, 1]).astype('int32')

import pdb; pdb.set_trace()
# # train model
# batch_size = 100
# max_iterations = 100000
# num_train = x_train.shape[0]

# for i in range(max_iterations):
#     idx = rng.permutation(num_train)
#     x_batch = x_train[idx[0:batch_size]]
#     y_batch = labels_train[idx[0:batch_size]]

#     for j in range(x_batch.shape[0]):
#         # print "%d of %d" % (j, x_batch.shape[0])
#         # randomly flip images
#         if rng.rand() > 0.5:
#             x_batch[j] = x_batch[j, :, ::-1, :]
#             # import pdb; pdb.set_trace()

#     cost = network['backprop'](x_batch, 1, y_batch)
#     # cost = network['backprop'](x_batch, y_batch)
#     if i % 1000 == 0:
#         print i
#         print "\t%f" % cost

#     if i % 5000 == 0:
#         predictions_test = network['prediction'](x_test, 1)
#         # predictions_test = network['prediction'](x_test)
#         accuracy = numpy.mean(predictions_test == labels_test)
#         print "\taccuracy: %.5f" % accuracy
#         print

# cost = network['backprop'](x_batch, 1, y_batch)
# # cost = network['backprop'](x_batch, y_batch)
# predictions_test = network['prediction'](x_test, 1)
# # predictions_test = network['prediction'](x_test)
# accuracy = numpy.mean(predictions_test == labels_test)
# print "%f" % cost
# print "accuracy: %.5f" % accuracy
# print
