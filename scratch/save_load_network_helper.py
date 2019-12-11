"""Saving and loading network helper."""
# import argparse
import numpy
# import pickle
# from sklearn.externals import joblib
# import helpers.process_hantman_mat as process_hantman_mat
import theano
# import theano.tensor as T
# import layers.layers as layers
# import time
# import optimizers.rmsprop as rmsprop
# import optimizers.adam as adam
# import optimizers.optim_helpers as optim_helpers
# import layers.lstm as lstm
# import math
# from scipy import signal
# import helpers.plot_helper as plot_helper
# import helpers.paths as paths
# import models.hantman_lstm as hantman_lstm


def create_test_data(opts):
    """Create some sample data for testing."""
    # mini batch test data
    x_t = numpy.zeros(
        (opts['seq_length'], opts['mini_batch_size'], opts['num_input']),
        dtype=theano.config.floatX)
    y_t = numpy.zeros(
        (opts['seq_length'], opts['mini_batch_size'], opts['num_classes']),
        dtype=theano.config.floatX)

    init_h = numpy.zeros(
        (opts['mini_batch_size'], opts['num_hidden']),
        dtype=theano.config.floatX)

    init_c = numpy.zeros(
        (opts['mini_batch_size'], opts['num_hidden']),
        dtype=theano.config.floatX)

    # full sequence test data
    x_seq = numpy.zeros(
        (opts['seq_length'], opts['num_input']),
        dtype=theano.config.floatX)
    y_seq = numpy.zeros(
        (opts['seq_length'], opts['num_classes']),
        dtype=theano.config.floatX)

    h_seq = numpy.zeros(
        (opts['num_hidden'],),
        dtype=theano.config.floatX)

    c_seq = numpy.ones(
        (opts['num_hidden'],),
        dtype=theano.config.floatX)

    data = dict()
    data['x_t'] = x_t
    data['y_t'] = y_t
    data['init_h'] = init_h
    data['init_c'] = init_c
    data['x_seq'] = x_seq
    data['y_seq'] = y_seq
    data['h_seq'] = h_seq
    data['c_seq'] = c_seq

    return data

# print network['cost'](data['x_t'], data['y_t'], data['init_h'], data['init_c'])
# print network['predict_seq'](data['x_seq'], data['h_seq'], data['c_seq'])

# print "changing network"
# size = network['params'][15].get_value().shape
# network['params'][15].set_value(numpy.ones(
#     size, dtype=theano.config.floatX))

# print network['cost'](data['x_t'], data['y_t'], data['init_h'], data['init_c'])
# print network['predict_seq'](data['x_seq'], data['h_seq'], data['c_seq'])