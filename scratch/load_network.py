"""Test network saving/loading."""
# additionally test the creation of a lstm that is trained on sequence chunks
# but can be tested on full videos.
import theano
import theano.tensor as T
import numpy
# import layers.lstm as lstm
import helpers.sample_data_helper as sample_data_helper
# import layers.layers as layers
import time
# import optimizers.adam as adam
# import optimizers.optim_helpers as optim_helpers
import tests.load_network_helper as load_network_helper

rng = numpy.random.RandomState(123)
opts = dict()
opts['rng'] = rng
opts['num_input'] = 10
opts['num_hidden'] = 20
opts['mini_batch_size'] = 5
opts['seq_length'] = 7
opts['num_classes'] = 3
opts['num_samples'] = 10000
opts['weightdecay'] = 0.0001
opts['decay_step'] = 0
opts['decay_rate'] = 0.5
opts['total_iterations'] = 20000
base_lr = 0.0001

num_train = opts['num_samples']
num_test = opts['num_samples']
seq_length = opts['seq_length']

network = load_network_helper.create_network(rng, opts)

# create some test data
centers = sample_data_helper.create_centers(
    rng, opts['num_classes'], opts['num_input'])

# creates data in [samples x seq_length x feature dims] size
train_data, train_labels = sample_data_helper.create_sample_sequences(
    rng, centers, opts['num_classes'], opts['num_samples'], opts['seq_length'],
    opts['num_input'])
test_data, test_labels = sample_data_helper.create_sample_sequences(
    rng, centers, opts['num_classes'], opts['num_samples'], opts['seq_length'],
    opts['num_input'])
long_seq, long_labels = sample_data_helper.create_sample_sequences(
    rng, centers, opts['num_classes'],
    opts['num_samples'], opts['seq_length'] * 2,
    opts['num_input'])

# create new labels, even/odd
train_even_odd = numpy.zeros((num_train, seq_length),
                             dtype='int64')
for i in range(num_train):
    for j in range(seq_length):
        train_even_odd[i, j] = train_labels[i, range(j + 1)].sum() % 2

test_even_odd = numpy.zeros((num_test, seq_length),
                            dtype='int64')
for i in range(num_test):
    for j in range(seq_length):
        test_even_odd[i, j] = train_labels[i, range(j + 1)].sum() % 2


# reshape the test data
test_x = numpy.zeros((seq_length, num_test, opts['num_input']),
                     dtype=theano.config.floatX)
for i in range(num_test):
    for j in range(seq_length):
        for k in range(opts['num_input']):
            test_x[j, i, k] = test_data[i, j, k]
test_even_odd = test_even_odd.transpose()

computed_cost = 0
t0 = time.time()
for i in range(opts['total_iterations']):
    # load batch
    idx = rng.permutation(opts['num_samples'])
    # print idx[1:opts['mini_batch_size']]
    examples_org = train_data[idx[1:opts['mini_batch_size'] + 1], :]
    # print examples_org.shape
    labels = train_even_odd[idx[1:opts['mini_batch_size'] + 1], :]

    examples = numpy.zeros(
        (seq_length, opts['mini_batch_size'], opts['num_input']),
        dtype=theano.config.floatX)
    for a in range(opts['mini_batch_size']):
        for b in range(seq_length):
            for c in range(opts['num_input']):
                examples[b, a, c] = examples_org[a, b, c]

    computed_cost = network['backprop'](examples, labels.transpose())

    # shared_train.set_value(examples)
    # computed_cost = f_grad_shared_given(labels)
    # f_update(learning_rate)
    # f_update()

    if i % 100 == 0:
        print "%d iters: %f" % (i, computed_cost)
        for g in network['params']:
            print "\t%s: %f" % (g.name, g.get_value().mean())
        # print "\tt: %f" % lr_update[1][0].get_value()
        # print "\tlearning rate: %f" % learning_rate.eval()
        # print "\tlearning rate: %f" % learning_rate.get_value()
t1 = time.time()
print computed_cost
print "total time: %f" % (t1 - t0)
print "time per iter: %f" % ((t1 - t0) / opts['total_iterations'])

predictions = theano.function(network['vars'], network['prediction'])

error_sum = network['error_sum']

num_missed = 0
for i in range(num_test / opts['mini_batch_size']):
    start_idx = (i - 1) * opts['mini_batch_size']
    end_idx = i * opts['mini_batch_size']
    num_missed += error_sum(test_x[:, range(start_idx, end_idx)],
                            test_even_odd[:, range(start_idx, end_idx)])

print num_missed
print num_missed * 1.0 / num_test

# create the full sequence checker
predict_sequence, error_sequence =\
    load_network_helper.create_prediction_network(rng, opts, network)
# long_seq[0, :, :], long_labels[0, :, :]
