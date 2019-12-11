"""Train a convnet on cifar 10 data.

From: https://github.com/benanne/theano-tutorial
"""
import theano
# import theano.tensor as T
import numpy
import helpers.load_cifar as load_cifar
import models.vgg_network as vgg_network
# import layers.losses as losses

# some globals.
rng = numpy.random.RandomState(123)
eps = numpy.finfo(numpy.float32).eps

# load data
x_train, labels_train, x_test, labels_test = load_cifar.cifar10(
    dtype=theano.config.floatX, grayscale=False)

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
network = vgg_network.create_network(opts)

# create the backprop and solver
network = vgg_network.create_loss_layer(opts, network)
network = vgg_network.create_optimizer(opts, network)

# train model
batch_size = 100
max_iterations = 100000
num_train = x_train.shape[0]

for i in range(max_iterations):
    idx = rng.permutation(num_train)
    x_batch = x_train[idx[0:batch_size]]
    y_batch = labels_train[idx[0:batch_size]]

    for j in range(x_batch.shape[0]):
        # print "%d of %d" % (j, x_batch.shape[0])
        # randomly flip images
        if rng.rand() > 0.5:
            x_batch[j] = x_batch[j, :, ::-1, :]
            # import pdb; pdb.set_trace()

    cost = network['backprop'](x_batch, 1, y_batch)
    # cost = network['backprop'](x_batch, y_batch)
    if i % 1000 == 0:
        print i
        print "\t%f" % cost

    if i % 5000 == 0:
        predictions_test = network['prediction'](x_test, 1)
        # predictions_test = network['prediction'](x_test)
        accuracy = numpy.mean(predictions_test == labels_test)
        print "\taccuracy: %.5f" % accuracy
        print

cost = network['backprop'](x_batch, 1, y_batch)
# cost = network['backprop'](x_batch, y_batch)
predictions_test = network['prediction'](x_test, 1)
# predictions_test = network['prediction'](x_test)
accuracy = numpy.mean(predictions_test == labels_test)
print "%f" % cost
print "accuracy: %.5f" % accuracy
print
