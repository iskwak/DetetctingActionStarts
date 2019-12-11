"""Helper functions for load_network.py. Makes it easier to read."""
import theano
import theano.tensor as T
import numpy
import layers.lstm as lstm
import layers.layers as layers
import time
import optimizers.optim_helpers as optim_helpers
import optimizers.adam as adam

base_lr = 0.0001


def create_network(rng, opts):
    """Create the base network, with gradient functions."""
    names = {'gates': 'layer1_gates', 'transform': 'layer1_transform'}
    lstm_node = lstm.LSTM(opts, names)
    lstm_node.create_weights()

    # he transformation to convert h_t into y_t
    w = layers.create_weight_matrix(
        rng, opts['num_hidden'], opts['num_classes'], name='embed_w')

    b = theano.shared(
        value=numpy.zeros((opts['num_classes'],), dtype=theano.config.floatX),
        name="embed_b", borrow=True)

    # create a single step for softmax.
    def lstm_step(x_t, yhat_t, h_tm1, c_tm1):
        """Lambda function for scan."""
        return lstm_node.lstm_step_softmax(x_t, yhat_t, h_tm1, c_tm1, w, b)

    x_t = T.tensor3('x_t')
    yhat_t = T.matrix('yhat_t', 'int64')
    params = lstm_node.params.values() + [w, b]

    [h_t, c_t, y_t, py_t, pyt_cost, is_correct], _ = theano.scan(
        lstm_step,
        sequences=[x_t, yhat_t],
        outputs_info=[lstm_node.init_h, lstm_node.init_c,
                      None, None, None, None]
    )

    # create the cost function
    cost_op = pyt_cost.sum(axis=0).mean()

    # add weight decay
    weight_decay = theano.shared(numpy.asarray(opts['weightdecay'],
                                               dtype=theano.config.floatX))

    # add in the weight decay
    l2_reg = 0
    for param in params:
        if '_w' in param.name:
            print param.name
            l2_reg += (param**2).sum()
    # apply the weighting
    l2_reg *= weight_decay

    # full cost op
    cost_op = l2_reg + cost_op

    # create the gradient op for the cost
    grads_ops = T.grad(cost_op, params)

    # clip only the lstm parameters, and only the weights not the biases
    for i in range(len(grads_ops)):
        if 'lstm_w' in params[i].name or 'lstm_w' in params[i].name:
            print params[i].name
            grads_ops[i] = T.clip(grads_ops[i], -15., 15.)

    print "Creating adam optimizer..."
    tic = time.time()
    if opts['decay_step'] > 0:
        lr_update = optim_helpers.step_lr_update(decay_rate=opts['decay_rate'],
                                                 decay_step=opts['decay_step'])
    else:
        lr_update = None
    # print lr_update
    # print lr_update[0]

    backprop, learning_rate, grad_vals = adam.create_adam_updates(
        params, grads_ops, x_t, yhat_t, cost_op,
        lr_update=lr_update, alpha=base_lr)
    # backprop, learning_rate, grad_vals = rmsprop.rms_prop(
    #     params, grads_ops, x_t_sym, y_t_sym, cost_op,
    #     lr_update=None, alpha=base_lr)
    toc = time.time()
    print "... took %d seconds" % (toc - tic)

    print "Creating error_sum function..."
    tic = time.time()
    error_sum = theano.function([x_t, yhat_t], T.neq(
        is_correct.sum(axis=0), 0).sum())
    toc = time.time()
    print "... took %d seconds" % (toc - tic)

    network = dict()
    network['lstm'] = lstm_node
    network['backprop'] = backprop
    network['learning_rate'] = learning_rate
    network['error_sum'] = error_sum
    network['grads_ops'] = grads_ops
    network['cost_op'] = cost_op
    network['prediction'] = y_t
    network['w_b'] = [w, b]
    network['params'] = lstm_node.params.values() + [w, b]
    network['vars'] = [x_t, yhat_t]

    return network


def create_prediction_network(rng, opts, network):
    """Create a network for predicting a full sequence."""
    # create the prediction network
    init_h, init_c = network['lstm'].create_seq_inits()

    x_data_t = T.matrix('x_t')
    yhat_data_t = T.vector('yhat_t', 'int64')

    w, b = network['w_b']

    # create a single step for softmax.
    def lstm_step_softmax(x_t, yhat_t, prev_h, prev_c):
        """Create an example softmax function to work with theano.scan."""
        return network['lstm_node'].single_seq_step(
            x_t, yhat_t, prev_h, prev_c, w, b)

    [h_t, c_t, py_t, pyt_cost, y_t, is_correct], _ = theano.scan(
        lstm_step_softmax,
        sequences=[x_data_t, yhat_data_t],
        outputs_info=[init_h, init_c, None, None, None, None]
    )

    print "Creating functions..."
    tic = time.time()
    error_sum = theano.function([x_data_t, yhat_data_t], T.neq(
        is_correct.sum(axis=0), 0).sum())

    predictions = theano.function([x_data_t, yhat_data_t],
                                  y_t)
    toc = time.time()
    print "... took %d seconds" % (toc - tic)
    return predictions, error_sum
