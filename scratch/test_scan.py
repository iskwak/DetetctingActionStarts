import theano
import theano.tensor as T
import numpy
import time

x = T.tensor3('x')


multi_scan, mult_scan_updates = theano.scan(
    lambda mat: mat,
    sequences=[x],
    name='multi_scan',
    n_steps=x.shape[0]
)

multi_func = theano.function([x], multi_scan)


num_samples = 3
seq_length = 4
num_dims = 5
x_vals = numpy.zeros(
    (num_samples, seq_length, num_dims),
    dtype=theano.config.floatX
)

for i in range(num_samples):
    x_vals[i] = x_vals[i] + i


loop_vals = numpy.ones(
    (num_samples, seq_length, num_dims),
    # rng.uniform(size=(dim_hid, dim_in)),
    dtype=theano.config.floatX
)

val = 0
for j in range(num_samples):
    for i in range(seq_length):
        loop_vals[j, i] = loop_vals[j, i] * val
        val = val + 1


# x = T.tensor3('x')
seq_var = T.tensor3('seq_var')
x = T.matrix('x')

# first goal, to get the rows of the 0'th index


theano.config.optimizer = 'fast_run'


def get_rows(x_t, accum):
    x_t = x_t * accum[0]
    accum = accum + 1
    return x_t, accum

get_rows_op, _ = theano.scan(
    get_rows,
    sequences=x,
    outputs_info=[None, theano.shared(numpy.zeros((1), dtype='float32'))]
)
get_rows_func = theano.function([x], get_rows_op)
#print get_rows_func(loop_vals[0])

# get a "sample"
get_sample_op, _ = theano.scan(
    get_rows,
    sequences=seq_var,
    outputs_info=[None, theano.shared(numpy.zeros((1), dtype='float32'))]
)
get_sample_func = theano.function([seq_var], get_sample_op)
#print get_sample_func(loop_vals)

# double loop


def get_sample(x_sample, row_accum, sample_accum):
    # setup the double loop
    [x_t, row_accum_op], _ = theano.scan(
        get_rows,
        sequences=x_sample,
        outputs_info=[None, row_accum[-1]]
        # outputs_info=[None, theano.shared(numpy.zeros((1), dtype='float32'))]
    )
    sample_accum = sample_accum + 1
    return x_t, row_accum_op, sample_accum

row_accums = theano.shared(numpy.zeros((seq_length,1), dtype='float32'))
get_sample_op, _ = theano.scan(
    get_sample,
    sequences=[seq_var],
    outputs_info=[None,
                  row_accums,
                  theano.shared(numpy.zeros((1), dtype='float32'))
                  ]
)
get_sample_func = theano.function([seq_var], get_sample_op)
print get_sample_func(loop_vals)


