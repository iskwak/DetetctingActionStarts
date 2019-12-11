import theano
import theano.tensor as T
import layers.layers as layers
import numpy

# hack to help testing in a repl...
reload(layers)

x = T.matrix('x')
prev_h = T.matrix('prev_h')
prev_c = T.matrix('prev_c')
rng = numpy.random.RandomState(1234)
dims = 4


# create lstm_sum_params
i2h_w_values = 	w_values = numpy.asarray(
					rng.uniform(
						low=-numpy.sqrt(1.0 / (dims + dims)),
						high=numpy.sqrt(1.0 / (dims + dims)),
						size=(dims, dims)
					),
					dtype=theano.config.floatX
				)
h2h_w_values = w_values = numpy.asarray(
					rng.uniform(
						low=-numpy.sqrt(1.0 / (dims + dims)),
						high=numpy.sqrt(1.0 / (dims + dims)),
						size=(dims, dims)
					),
					dtype=theano.config.floatX
				)
in_gate = layers.lstm_sum_params(rng, dims, name='in_gate',
								 i2h_w_values=i2h_w_values,
								 h2h_w_values=h2h_w_values)

rng = numpy.random.RandomState(1234)

# test the creation of lstm_sum_params
params = layers.lstm_sum_params(rng, dims, name='in_gate', scale=None)


#lstm_dict = layers.create_lstm_params(rng, dims)


#next_h, next_c = layers.create_lstm_gates(x, prev_h, prev_c, lstm_dict)


#lstm_func = theano.function([x,prev_h,prev_c], [next_h,next_c])


#seq = numpy.zeros((1,10), dtype=theano.config.floatX)
#init_h = numpy.zeros((1,10), dtype=theano.config.floatX)
#init_c = numpy.zeros((1,10), dtype=theano.config.floatX)

#beep = lstm_func(seq, init_h, init_c)

# W = create_weight_matrix(rng, dims, dims,
# 						 name="layer", W=None,
# 						 initialization='xavier',
# 						 scale=None)

# b = theano.shared(value=numpy.zeros((dims,),
# 	dtype=theano.config.floatX), name="layer_b", borrow=True)

# (fc_output, param_W, param_b) = create_fully_connected_layer(
# 	rng, x, dims, dims)

# next_h, next_c, lstm_params = layers.create_lstm_node(
# 	rng, x, prev_h, prev_c, dims)

