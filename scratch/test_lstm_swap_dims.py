import theano
import theano.tensor as T
import layers.layers as layers
import numpy
import time
import tests.test_sample_data_helper as test_sample_data_helper
import tests.test_threaded_loading_helper as test_threaded_loading_helper

num_dims = 10
num_rnn = 6
num_train = 10000
num_test = 1000
mini_batch_size = 100
seq_length = 2
num_classes = 3
num_samples = 10
rng = numpy.random.RandomState(1234)
eps = numpy.finfo(numpy.float32).eps


x_t_sym = T.tensor3('x_t_sym')
y_t_sym = T.matrix('y_t_sym', dtype='int64')

lr = T.scalar('lr')


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


lstm_dict = layers.create_lstm_params(rng, num_dims, num_rnn)

# the transformation to convert h_t into y_t
w_values = numpy.random.randint(10, size=(num_rnn, num_classes))
w_values = w_values.astype('float32')
w = layers.create_weight_matrix(
    rng, num_rnn, num_classes, name='W', w_values=w_values)

b = theano.shared(
    value=numpy.zeros((num_classes,), dtype=theano.config.floatX),
    name="b", borrow=True)


def lstm_step(x_t, yhat_t, h_tm1, c_tm1):
    [h_t, c_t] = layers.create_lstm_gates(x_t, h_tm1, c_tm1, lstm_dict)

    # P(y_t)
    py_t = T.nnet.softmax(T.dot(h_t, w) + b)

    # error for each sequence at time t
    pyt_cost = -T.log(py_t)[T.arange(yhat_t.shape[0]), yhat_t]

    # predicted class for the sequence at time t
    y_t = T.argmax(py_t, axis=1)
    is_correct = T.neq(y_t, yhat_t)

    return h_t, c_t, y_t, py_t, pyt_cost, is_correct

# seq_predict = T.argmax(T.nnet.softmax(y_t), axis=1)

# seq_scores = -T.log(p_y_t)
# seq_cost = -T.log(p_y_t)[T.arange(y.shape[0]),y].sum()
# seq_predict = T.argmax(T.nnet.softmax(y_t), axis=1)

# h_t, c_t, y_t, define the outputs of the LSTM at a single interval for all
# samples in a batch.
init_h = T.alloc(numpy.cast[theano.config.floatX](0), mini_batch_size, num_rnn)
init_c = T.alloc(numpy.cast[theano.config.floatX](0), mini_batch_size, num_rnn)

[h_t, c_t, y_t, py_t, pyt_cost, is_correct], _ = theano.scan(
    lstm_step,
    sequences=[x_t_sym, y_t_sym],
    outputs_info=[init_h, init_c, None, None, None, None]
)

cost_op = pyt_cost.sum(axis=0).mean()

params = lstm_dict.values() + [W, b]

print "calculating gradient graph"
tic = time.time()
grads_ops = T.grad(cost_op, params)
# grads_ops = [T.clip(g, -5., -5.) for g in grads_ops]
toc = time.time()
print "T.grad: %d seconds" % (toc - tic)

# shared memory for the gradients
shared_grads = [theano.shared(numpy.zeros(p.get_value().shape,
                                          dtype=theano.config.floatX),
                              name='%s_grad' % p.name)
                for p in params]


print "Creating forward pass function graph"
t0 = time.time()
# update rule for the shared gradients, this just stores calculated gradients
# into the shared memory space. useful for debuggings
shared_grad_updates = [(gs, g) for gs, g in zip(shared_grads, grads_ops)]
f_grad_shared = theano.function([x_t_sym, y_t_sym], cost_op,
                                updates=shared_grad_updates,
                                name='sgd_f_grad_shared')
t1 = time.time()
print "f_grad_shared: %d" % (t1 - t0)

# shared_train = theano.shared(
#     numpy.zeros((mini_batch_size, num_dims), dtype=theano.config.floatX),
#     name='shared_train', borrow=True)
# f_grad_shared_given = theano.function([y], cost_op,
#                                       updates=shared_grad_updates,
#                                       name='sgd_f_grad_shared',
#                                       givens=[(mini_batch_var, shared_train)]
#                                       )

t0 = time.time()
# update rule for the parameters. This will use the calculated gradients to
# update the parameters
params_update = [(p, p - lr * g) for p, g in zip(params, shared_grads)]
f_update = theano.function(
    [lr], [], updates=params_update, name='sgd_f_update')
t1 = time.time()
print "f_update: %d" % (t1 - t0)


errors = theano.function([x_t_sym, y_t_sym], T.neq(
    is_correct.sum(axis=0), 0).mean())
# costs = theano.function([x_t_sym, y_t_sym], [
#                         py_t, pyt_cost, pyt_cost.sum(axis=0),
#                         is_correct,
#                         T.neq(is_correct.sum(axis=0), 0).mean()
#                         ])
# classify_sequence = theano.function([x_t_sym, y_t_sym], [y_t])


# create some data
centers = test_sample_data_helper.create_centers(rng, num_classes, num_dims)
# train_x_org, train_y = test_sample_data_helper.create_sample_sequences(
train_x, train_y = test_sample_data_helper.create_sample_sequences(
    rng, centers, num_classes, num_train, seq_length, num_dims)

test_x, test_y = test_sample_data_helper.create_sample_sequences(
    rng, centers, num_classes, num_test, seq_length, num_dims)

# reshape the data
# train_x = numpy.zeros((seq_length, num_train, num_dims),
#                       dtype=theano.config.floatX)
# for i in range(num_train):
#     for j in range(seq_length):
#         for k in range(num_dims):
#             train_x[j, i, k] = train_x_org[i, j, k]
# train_y = train_y.transpose()


loader = test_threaded_loading_helper.DataLoaderPool(
    train_x, train_y, mini_batch_size)

total_iterations = 10000

computed_cost = 0
t0 = time.time()
learning_rate = 0.001
for i in range(total_iterations):
    # load batch
    (examples_org, labels) = loader.get_minibatch()
    examples = numpy.zeros(
        (seq_length, mini_batch_size, num_dims), dtype=theano.config.floatX)
    for a in range(mini_batch_size):
        for b in range(seq_length):
            for c in range(num_dims):
                examples[b, a, c] = examples_org[a, b, c]

    computed_cost = f_grad_shared(examples, labels.transpose())

    # shared_train.set_value(examples)
    # computed_cost = f_grad_shared_given(labels)
    f_update(learning_rate)

    if i % 100 == 0:
        print "%d iters: %f" % (i, computed_cost)
t1 = time.time()
print computed_cost
print "total time: %f" % (t1 - t0)
print "time per iter: %f" % ((t1 - t0) / total_iterations)
del loader
