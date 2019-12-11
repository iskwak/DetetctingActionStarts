"""Testing f(x) and f(x+h) updates."""
import theano
import theano.tensor as T
import numpy

nfl/
dim_out = 10
dim_in = 10

# rng = numpy.random.RandomState(1234)
w_values = numpy.identity(
    dim_out,
    dtype=theano.config.floatX
)


w = theano.shared(value=w_values, name="W", borrow=True)
b_values = numpy.zeros((dim_out,), dtype=theano.config.floatX)
b = theano.shared(value=b_values, name="%b", borrow=True)

x = T.matrix('x')
# y = T.matrix('y', dtype='int64')

func_sym = T.dot(x, w) + b
cost = (func_sym**2).sum()

grad = T.grad(cost, [w, b])
