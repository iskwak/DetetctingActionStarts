import theano
import numpy


# from:
# http://deeplearning.net/software/theano/tutorial/aliasing.html

# borrow=True should allow the use of a numpy array to access the contents
# of a memory space on the GPU.
# this example should print out:
# [ 1.  1.]
# [ 1.  1.]
# [ 2.  2.]
#
# However doesn't seem to work.
np_array = numpy.ones(2, dtype='float32')

s_default = theano.shared(np_array)
s_false = theano.shared(np_array, borrow=False)
s_true = theano.shared(np_array, borrow=True)

np_array += 1  # now it is an array of 2.0 s

print('Test constructor values')
print(s_default.get_value())
print(s_false.get_value())
print(s_true.get_value())
print('final array should be [2. 2.]... but doesn\'t seem to be\n')
# it doesn't seem that s_true has updated values. It is unclear what
# borrow=True is doing.

# next, try to use get_value to have a handle to the numpy representation of the
# gpu data.
s = theano.shared(np_array)

v_false = s.get_value(borrow=False)  # N.B. borrow default is False
v_true = s.get_value(borrow=True)

print('Test get_value values')
print('v_false: %s' % v_false)
v_false = v_false + 1
print('v_false modified: %s' % s.get_value())
print('v_true: %s' % v_true)
v_true = v_true + 1
print('v_true modified: %s' % s.get_value())
print('In this case the output should be [3. 3.]... but is not\n')

del v_true
del v_false
del s
del np_array
del s_default
del s_false
del s_true


# the final option to modifying memory of shared variables on the gpu is with
# set_value.
# here I deviate from the examples from theano to make sure that set_value does
# an in place replacement of values in the GPU

# roughly 3600 MiB (seems to be more... maybe because of some overhead?). The
# GPU only has 6143MiB, so it will not be able to create another array to copy
# data into.
zeros = numpy.zeros((100000,12000), dtype=theano.config.floatX)
ones = numpy.ones((100000,12000), dtype=theano.config.floatX)

# create the shared variable... ignoring borrow
gpu_data = theano.shared(value=zeros)
print('gpu_data before set')
print(gpu_data.get_value())
print('gpu_data after set')
gpu_data.set_value(ones)
print(gpu_data.get_value())
print('gpu_data should now be all ones')

del zeros
del ones
del gpu_data