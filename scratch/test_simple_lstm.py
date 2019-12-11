import theano
import theano.tensor as T
import layers.layers as layers
import numpy
# import test_lstm_helper
import tests.test_threaded_loading_helper as test_threaded_loading_helper
import time
import helpers.sample_data_helper as sample_data_helper
import optimizers.adam as adam
import optimizers.optim_helpers as optim_helpers

# hack to help testing in a repl...
# reload(layers)
# # reload(test_lstm_helper)
# reload(test_threaded_loading_helper)
# reload(sample_data_helper)


total_iterations = 10000
num_dims = 20
num_rnn = 10
num_train = 10000
num_test = 1000
mini_batch_size = 100
seq_length = 20
num_classes = 2
num_centers = 10
rng = numpy.random.RandomState(1234)
eps = numpy.finfo(numpy.float32).eps


x_t_sym = T.tensor3('x_t_sym')
y_t_sym = T.matrix('y_t_sym', dtype='int64')
lr = T.scalar('lr')


def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer
    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize
    Notes
    -----
    For more information, see [ADADELTA]_.
    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % p.name)
                    for p in tparams]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % p.name)
                   for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % p.name)
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    print running_grads2
    print rg2up

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


lstm_dict = layers.create_lstm_params(rng, num_dims, num_rnn)

# the transformation to convert h_t into y_t
w_values = numpy.random.randint(10, size=(num_rnn, num_classes))
w_values = w_values.astype('float32')
w = layers.create_weight_matrix(
    rng, num_rnn, num_classes, name='embed', w_values=w_values)

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

params = lstm_dict.values() + [w, b]

print "calculating gradient graph"
tic = time.time()
grads_ops = T.grad(cost_op, params)
# print grads_ops
# grads_ops = [T.clip(g, -15., 15.) for g in grads_ops]
# clip only the lstm parameters, and only the weights not the biases
for i in range(len(grads_ops)):
    if 'i2h_W' in params[i].name or 'h2h_W' in params[i].name:
        grads_ops[i] = T.clip(grads_ops[i], -15., 15.)
# print grads_ops
toc = time.time()
print "T.grad: %d seconds" % (toc - tic)

# shared memory for the gradients
shared_grads = [theano.shared(numpy.zeros(p.get_value().shape,
                                          dtype=theano.config.floatX),
                              name='%s_grad' % p.name)
                for p in params]


print "Creating adam update function"
tic = time.time()
# f_grad_shared, f_update = adadelta(
#     lr, params, grads_ops, x_t_sym, y_t_sym, cost_op)
# f_grad_shared, f_update = adam.create_adam_updates(
#     params, grads_ops, x_t_sym, y_t_sym, cost_op)
lr_update = optim_helpers.step_lr_update()
lr_update = None
f_grad_shared, learning_rate, grad_vals = adam.create_adam_updates(
    params, grads_ops, x_t_sym, y_t_sym, cost_op,
    lr_update=lr_update)
# updates = adam.create_adam_updates(params, grads_ops,
#                                    x_t_sym, y_t_sym, cost_op)
# f_grad_shared = theano.function([x_t_sym, y_t_sym], cost_op, updates=updates,
#                                name='adam_f_grad_shared')
toc = time.time()
print "adam: %d seconds" % (toc - tic)

# print "Creating forward pass function graph"
# t0 = time.time()
# # update rule for the shared gradients, this just stores calculated gradients
# # into the shared memory space. useful for debuggings
# shared_grad_updates = [(gs, g) for gs, g in zip(shared_grads, grads_ops)]
# f_grad_shared = theano.function([x_t_sym, y_t_sym], cost_op,
#                                 updates=shared_grad_updates,
#                                 name='sgd_f_grad_shared')
# t1 = time.time()
# print "f_grad_shared: %d" % (t1 - t0)

# shared_train = theano.shared(
#     numpy.zeros((mini_batch_size, num_dims), dtype=theano.config.floatX),
#     name='shared_train', borrow=True)
# f_grad_shared_given = theano.function([y], cost_op,
#                                       updates=shared_grad_updates,
#                                       name='sgd_f_grad_shared',
#                                       givens=[(mini_batch_var, shared_train)]
#                                       )

# t0 = time.time()
# # update rule for the parameters. This will use the calculated gradients to
# # update the parameters
# params_update = [(p, p - lr * g) for p, g in zip(params, shared_grads)]
# f_update = theano.function(
#     [lr], [], updates=params_update, name='sgd_f_update')
# t1 = time.time()
# print "f_update: %d" % (t1 - t0)


errors = theano.function([x_t_sym, y_t_sym], T.neq(
    is_correct.sum(axis=0), 0).mean())
# costs = theano.function([x_t_sym, y_t_sym], [
#                         py_t, pyt_cost, pyt_cost.sum(axis=0),
#                         is_correct,
#                         T.neq(is_correct.sum(axis=0), 0).mean()
#                         ])
# classify_sequence = theano.function([x_t_sym, y_t_sym], [y_t])


# create some data
centers = sample_data_helper.create_centers(rng, num_centers, num_dims)
# train_x_org, train_y = sample_data_helper.create_sample_sequences(
train_x, train_y = sample_data_helper.create_sample_sequences(
    rng, centers, num_centers, num_train, seq_length, num_dims)

test_x_org, test_y = sample_data_helper.create_sample_sequences(
    rng, centers, num_centers, num_test, seq_length, num_dims)


# create new labels, even/odd
train_even_odd = numpy.zeros((num_train, seq_length),
                             dtype='int64')
for i in range(num_train):
    for j in range(seq_length):
        train_even_odd[i, j] = train_y[i, range(j + 1)].sum() % 2

test_even_odd = numpy.zeros((num_test, seq_length),
                            dtype='int64')
for i in range(num_test):
    for j in range(seq_length):
        test_even_odd[i, j] = test_y[i, range(j + 1)].sum() % 2


# reshape the test data
test_x = numpy.zeros((seq_length, num_test, num_dims),
                     dtype=theano.config.floatX)
for i in range(num_test):
    for j in range(seq_length):
        for k in range(num_dims):
            test_x[j, i, k] = test_x_org[i, j, k]
test_even_odd = test_even_odd.transpose()

# reshape the data
# train_x = numpy.zeros((seq_length, num_train, num_dims),
#                       dtype=theano.config.floatX)
# for i in range(num_train):
#     for j in range(seq_length):
#         for k in range(num_dims):
#             train_x[j, i, k] = train_x_org[i, j, k]
# train_y = train_y.transpose()


loader = test_threaded_loading_helper.DataLoaderPool(
    train_x, train_even_odd, mini_batch_size)


computed_cost = 0
t0 = time.time()
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
    # f_update(learning_rate)
    # f_update()

    if i % 100 == 0:
        print "%d iters: %f" % (i, computed_cost)
        for g in grad_vals:
            print "\t%s: %f" % (g.name, g.get_value().mean())
        # print "\tt: %f" % lr_update[1][0].get_value()
        # print "\tlearning rate: %f" % learning_rate.eval()
        # print "\tlearning rate: %f" % learning_rate.get_value()
t1 = time.time()
print computed_cost
print "total time: %f" % (t1 - t0)
print "time per iter: %f" % ((t1 - t0) / total_iterations)
del loader


predictions = theano.function([x_t_sym, y_t_sym], y_t)


error_sum = theano.function([x_t_sym, y_t_sym], T.neq(
    is_correct.sum(axis=0), 0).sum())

num_missed = 0
for i in range(num_test / mini_batch_size):
    start_idx = (i - 1) * mini_batch_size
    end_idx = i * mini_batch_size
    num_missed += error_sum(test_x[:, range(start_idx, end_idx)],
                            test_even_odd[:, range(start_idx, end_idx)])

print num_missed
print num_missed * 1.0 / num_test
