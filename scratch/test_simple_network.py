import theano
import theano.tensor as T
import layers.layers as layers
import numpy
import tests.test_threaded_loading_helper as test_threaded_loading_helper
import time


# parameters
dim_in = 4096
dim_hid = 5
dim_out = 4
num_train = 100000
num_test = 1000
#num_classes = 4
num_classes = dim_out
learning_rate = 0.01

# create a two layer mlp
rng = numpy.random.RandomState(12)
eps = numpy.finfo(numpy.float32).eps

# input matrix
x = T.matrix('x')
# class variables
y = T.vector('y', dtype='int64')
# learning rate
lr = T.scalar(name='lr')

# layer 1
fc_layer1, layer1_W, layer1_b = layers.create_fully_connected_layer(
    rng, x, dim_in, dim_hid, name='layer1', activation=T.nnet.sigmoid)

# layer 2
fc_layer2, layer2_W, layer2_b = layers.create_fully_connected_layer(
    rng, fc_layer1, dim_hid, dim_out, name='layer2', activation=None)

# probability function
prob_ops = T.nnet.softmax(fc_layer2)
probs = theano.function([x], prob_ops)

# prediction function
predict_op = -T.log(T.nnet.softmax(fc_layer2)[T.arange(y.shape[0]), y])
predict = theano.function([x, y], predict_op)

prediction_op = T.argmax(T.nnet.softmax(fc_layer2), axis=1)
prediction = theano.function([x], prediction_op)


# cost function
cost_op = T.mean(predict_op)
cost = theano.function([x, y], cost_op)
#cost_given = theano.function([], cost_op)

# list of params
params = [layer1_W, layer1_b, layer2_W, layer2_b]

# gradient (list of grad ops)
grads_ops = T.grad(cost_op, wrt=params)

# shared memory for the gradients
shared_grads = [theano.shared(numpy.zeros(p.get_value().shape,
                                          dtype=theano.config.floatX),
                              name='%s_grad' % p.name)
                for p in params]

# update rule for the shared gradients, this just stores calculated gradients
# into the shared memory space. useful for debuggings
shared_grad_updates = [(gs, g) for gs, g in zip(shared_grads, grads_ops)]
f_grad_shared = theano.function([x, y], cost_op, updates=shared_grad_updates,
                                name='sgd_f_grad_shared')

# update rule for the parameters. This will use the calculated gradients to
# update the parameters
params_update = [(p, p - lr * g) for p, g in zip(params, shared_grads)]
f_update = theano.function(
    [lr], [], updates=params_update, name='sgd_f_update')


# create test data to make sure the neural network learns properly
# fake data
x_vals = numpy.zeros(
    (num_train, dim_in),
    dtype=theano.config.floatX
)
y_vals = numpy.ones(
    (num_train,),
    dtype='int64'
)

# randomly create centers for the classes
centers = numpy.asarray(
    rng.uniform(low=-5.0, high=5.0, size=(num_classes, dim_in)),
    dtype=theano.config.floatX
)

# training data
for i in range(num_classes):
    label = i
    start_idx = i * num_train / num_classes
    end_idx = (i + 1) * num_train / num_classes
    idx_range = range(start_idx, end_idx)
    y_vals[idx_range] = label * y_vals[idx_range]

    x_vals[idx_range] = rng.normal(
        loc=centers[i], size=(num_train / num_classes, dim_in))

# testing data
x_test = numpy.zeros(
    (num_test, dim_in),
    dtype=theano.config.floatX
)
y_test = numpy.ones(
    (num_test,),
    dtype='int64'
)

for i in range(num_classes):
    label = i
    start_idx = i * num_test / num_classes
    end_idx = (i + 1) * num_test / num_classes
    idx_range = range(start_idx, end_idx)
    y_test[idx_range] = label * y_test[idx_range]

    x_test[idx_range] = rng.normal(
        loc=centers[i], size=(num_test / num_classes, dim_in))


# begin training
shuffle_idx = range(num_train)
rng.shuffle(shuffle_idx)

mini_batch_size = 100
total_iterations = 10000

print 'creating data loader'
# create the data loader
loader = test_threaded_loading_helper.DataLoaderPool(
    x_vals, y_vals, mini_batch_size)

# create theano shared memory (on gpu memory) for storing features and labels
shared_train = theano.shared(
    numpy.zeros((mini_batch_size, dim_in), dtype=theano.config.floatX),
    name='shared_train', borrow=True)
shared_labels = theano.shared(
    numpy.zeros((mini_batch_size,), dtype='int64'),
    name='shared_labels')

f_grad_shared_given = theano.function([y], cost_op, updates=shared_grad_updates,
                                      name='sgd_f_grad_shared',
                                      givens=[(x, shared_train)]
                                      )


computed_cost = 0
t0 = time.time()
for i in range(total_iterations):
    # load batch
    (examples, labels) = loader.get_minibatch()

    #shared_labels.set_value(labels)
    #computed_cost = f_grad_shared(examples, labels)
    
    shared_train.set_value(examples)
    computed_cost = f_grad_shared_given(labels)
    f_update(learning_rate)

    if i % 100 == 0:
        print "%d iters: %f" % (i, computed_cost)
t1 = time.time()
print computed_cost
print "total time: %f" % (t1-t0)
print "time per iter: %f" % ((t1-t0)/total_iterations)
del loader


errors_op = T.mean(T.neq(prediction_op, y))
errors = theano.function([x, y], errors_op)

print errors(x_test, y_test)


cost_given = theano.function([y], cost_op, givens=[(x, shared_train)])
predict_given = theano.function([y], predict_op, givens=[(x, shared_train)])
prediction_given = theano.function(
    [], prediction_op, givens=[(x, shared_train)])
