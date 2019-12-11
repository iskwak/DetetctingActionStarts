"""Train a convnet on cifar 10 data.

based off of : https://github.com/benanne/theano-tutorial
"""
import argparse
import sys
import theano
# import theano.tensor as T
import numpy
import helpers.load_cifar as load_cifar
# import models.vgg_network as vgg_network
# import layers.losses as losses
import helpers.paths as paths
import os
from sklearn.externals import joblib
import models.vgg_network as vgg_network
import time
import models.convnetjs_cifar10 as convnetjs_cifar10

# some globals.
rng = numpy.random.RandomState(123)
eps = numpy.finfo(numpy.float32).eps

# default values for the module. Organized here to make it easier to change
# the defaults (if better values are ever found)
g_total_iterations = 10000
g_mini_batch_size = 100
g_learning_rate = 0.001
g_decay_step = 0
g_decay_rate = 0.5
g_weight_decay_val = 0.0001
g_out_dir = 'figs'
g_model_name = 'vgg_network'
g_num_classes = 10
g_use_bn = True
g_use_dropout = True
g_update_iter = 1000


def create_opts():
    """Create an opts dictionary."""
    opts = dict()
    opts['filename'] = ''
    opts['total_iterations'] = g_total_iterations
    opts['mini_batch_size'] = g_mini_batch_size
    opts['learning_rate'] = g_learning_rate
    opts['decay_step'] = g_decay_step
    opts['decay_rate'] = g_decay_rate
    opts['weight_decay'] = g_weight_decay_val
    opts['num_classes'] = g_num_classes
    opts['model_name'] = g_model_name
    opts['out_dir'] = g_out_dir
    opts['use_bn'] = g_use_bn
    opts['use_dropout'] = g_use_dropout
    opts['update_iter'] = g_update_iter
    return opts


def setup_opts(opts):
    """Setup default arguments for the arg parser.

    returns an opt dictionary with default values setup.
    """
    parser = argparse.ArgumentParser(description='Train an LSTM to classify '
                                     'mouse behavior data')
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='Location of the cifar data directory')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='output directory for figures')
    parser.add_argument('-i', '--iterations', default=g_total_iterations,
                        type=int, help='Max number of iterations')
    parser.add_argument('-b', '--batch', default=g_mini_batch_size, type=int,
                        help='Mini batch size')
    parser.add_argument('-r', '--learning_rate', default=g_learning_rate,
                        type=float, help='Learning rate')
    parser.add_argument('-s', '--decaystep', default=g_decay_step, type=float,
                        help='Learning rate decay step')
    parser.add_argument('-c', '--decayrate', default=g_decay_rate, type=float,
                        help='Learning rate decay rate')
    parser.add_argument('-w', '--weight_decay', default=g_weight_decay_val,
                        type=float, help='Weight decay lambda')
    parser.add_argument('-m', '--model', default=g_model_name,
                        type=str, help='Model to use')
    parser.add_argument('-u', '--update_iter', default=g_update_iter, type=int,
                        help='Output iterations')

    parser.add_argument('--dropout', dest='use_dropout',
                        action='store_true')
    parser.add_argument('--no-dropout', dest='use_dropout',
                        action='store_false')
    parser.set_defaults(use_bn=g_use_dropout)

    parser.add_argument('--batchnorm', dest='use_bn',
                        action='store_true')
    parser.add_argument('--no-batchnorm', dest='use_bn',
                        action='store_false')
    parser.set_defaults(use_bn=g_use_bn)

    args = parser.parse_args()

    opts['filename'] = args.filename
    opts['out_dir'] = args.out_dir
    opts['total_iterations'] = args.iterations
    opts['mini_batch_size'] = args.batch
    opts['learning_rate'] = args.learning_rate
    opts['decay_step'] = args.decaystep
    opts['decay_rate'] = args.decayrate
    opts['model'] = args.model
    opts['weight_decay'] = args.weight_decay
    opts['use_dropout'] = args.use_dropout
    opts['use_bn'] = args.use_bn
    opts['update_iter'] = args.update_iter

    return opts


def _flip_images(images):
    for j in range(images.shape[0]):
        # print "%d of %d" % (j, x_batch.shape[0])
        # randomly flip images
        if rng.rand() > 0.5:
            images[j] = images[j, :, ::-1, :]

    return images


def _process_network(opts, network, train_x, train_y, x_test, labels_test):
    # first decide if the is_train flag is needed for the network
    if opts['use_bn'] is False and opts['use_dropout'] is False:
        use_train = False
    else:
        use_train = True

    # create testing data
    mini_batch_size = opts['mini_batch_size']
    num_test = x_test.shape[0]
    idx = rng.permutation(num_test).astype('int32')
    test_x = x_test[idx[0:mini_batch_size]]
    test_y = labels_test[idx[0:mini_batch_size]]

    if use_train is False:
        train_args = [train_x, train_y]
        test_args = [test_x, test_y]
        test_predict_args = [test_x]
    else:
        train_args = [train_x, 1, train_y]
        test_args = [test_x, 0, test_y]
        test_predict_args = [x_test, 0]

    cost = network['backprop'](*train_args)

    t = int(network['lr_update']['params'][0].get_value())
    if t % opts['update_iter'] == 0:
        test_cost = network['cost'](*test_args)
        test_predicts = network['prediction'](*test_predict_args)
        accuracy = numpy.mean(test_predicts == labels_test)
    else:
        test_cost = None
        accuracy = None

    return cost, test_cost, accuracy


def _write_updates(opts, tic, network, train_cost, val_cost, val_acc):
    # helper function to write updates to disk

    t = int(network['lr_update']['params'][0].get_value())
    if t % opts['update_iter'] != 0:
        return None

    # update console
    print "%d iterations:" % t
    print "\t train loss; %f" % train_cost
    print "\t validation loss; %f" % val_cost
    print "\t validation accuracy; %f" % val_acc
    print "\t Seconds: %f" % (time.time() - tic)

    # update logs

    return


def train_network(opts, network, x_train, labels_train, x_test, labels_test):
    """Train the network."""
    num_train = opts['num_train']
    total_iterations = opts['total_iterations']
    mini_batch_size = opts['mini_batch_size']

    tic = time.time()
    round_tic = tic
    for i in range(total_iterations):
        idx = rng.permutation(num_train).astype('int32')

        # import pdb; pdb.set_trace()
        train_batch_x = x_train[idx[0:mini_batch_size]]
        train_batch_y = labels_train[idx[0:mini_batch_size]]

        train_batch_x = _flip_images(train_batch_x)

        # Process the network (backwards + testing forward if necessary)
        train_cost, val_cost, val_acc = _process_network(
            opts, network,
            train_batch_x, train_batch_y,
            x_test, labels_test)

        # if necessary, write the updates to disk
        _write_updates(opts, round_tic, network, train_cost, val_cost, val_acc)

        if i % opts['update_iter'] == 0:
            round_tic = time.time()

    idx = rng.permutation(num_train).astype('int32')
    train_batch_x = x_train[idx[0:mini_batch_size]]
    train_batch_y = labels_train[idx[0:mini_batch_size]]

    train_cost, val_cost, val_acc = _process_network(
        opts, network, train_batch_x, train_batch_y, x_test, labels_test)

    _write_updates(opts, network, train_cost, val_cost, val_acc)

    toc = time.time()
    print "Training took: %d seconds" % (toc - tic)

    return


def load_data(opts):
    """Load cifar data."""
    # load data
    x_train, labels_train, x_test, labels_test = load_cifar.cifar10(
        dtype=theano.config.floatX, grayscale=False,
        data_dir=opts['filename'])

    # DEBUG
    # x_train = x_train[0:10000]

    opts['num_train'] = x_train.shape[0]
    opts['num_test'] = x_test.shape[0]
    opts['img_dims'] = x_train.shape[1]
    # import pdb; pdb.set_trace()

    return x_train, labels_train, x_test, labels_test

if __name__ == "__main__":
    print sys.argv
    opts = create_opts()
    opts = setup_opts(opts)
    opts['rng'] = rng
    opts['eps'] = eps

    # create the output directory
    paths.create_dir(opts['out_dir'])
    paths.create_dir(opts['out_dir'] + "/predictions")
    paths.create_dir(opts['out_dir'] + "/plots")
    paths.create_dir(opts['out_dir'] + "/opts")
    paths.create_dir(opts['out_dir'] + '/grads')

    # save the command to a text file
    with open(os.path.join(opts['out_dir'], 'command.txt'), "w") as outfile:
        for i in range(len(sys.argv)):
            outfile.write(sys.argv[i] + " ")

    # save the options
    joblib.dump(opts, os.path.join(opts['out_dir'], 'opts', 'opts.npy'))

    # load the data
    x_train, labels_train, x_test, labels_test = load_data(opts)

    # create the network
    if opts['model'] == 'vgg_network':
        model = vgg_network
    elif opts['model'] == 'convnetjs':
        model = convnetjs_cifar10
    else:
        print "Unknown model: %s" % opts['model']
        sys.exit()
    # create base (forward) network
    network = model.create_network(opts)

    # create the backprop and solver
    network = model.create_loss_layer(opts, network)
    network = model.create_optimizer(opts, network)

    trained = train_network(
        opts, network, x_train, labels_train, x_test, labels_test)
