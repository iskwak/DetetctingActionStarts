"""Some code to play around with convolution sizes."""
import theano
import theano.tensor as T
import numpy
import layers.convolution as convolution

x = T.tensor4('x')

# batch size x in chan x width x height
input_shape = [10, 3, 48, 60]
data = numpy.zeros(input_shape, dtype='float32')

# out chan x in chan x width x height
filter_shape = [5, 3, 10, 15]
sub_sample = (2, 2)

# params to mess with
border_mode = 'half'

filters = theano.shared(
    value=numpy.zeros(filter_shape, dtype='float32'),
    name='filters', borrow=True)

conv_sym = T.nnet.conv2d(input=x, filters=filters,
                         filter_shape=filter_shape,
                         input_shape=None, border_mode=border_mode,
                         subsample=sub_sample)

conv = theano.function([x], conv_sym)

print "downsampled size"
print conv(data).shape
print convolution.compute_conv_dims(
    input_shape, filter_shape, border_mode=border_mode, sub_sample=sub_sample)

# next calculate the filter shape for upsampling
input_shape = conv(data).shape
target_shape = data.shape
border_mode = (0, 0)
sub_sample = (1, 1)
deconv_shape = convolution.compute_upscale_dims(
    target_shape, input_shape, sub_sample=None, border_mode=border_mode)

print "upsample filter shape"
print deconv_shape

# the filter shape is super confuing
# (current number of filters, target number of filters, width, height)
deconv_filters = theano.shared(
    value=numpy.zeros(deconv_shape, dtype='float32'),
    name='deconv', borrow=True)
deconv_sym = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
    x, filters=deconv_filters, filter_shape=deconv_shape,
    input_shape=target_shape, border_mode=border_mode)

deconv = theano.function([x], deconv_sym)

print deconv(conv(data)).shape
print data.shape
