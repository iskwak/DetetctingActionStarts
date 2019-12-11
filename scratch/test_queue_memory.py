# testing how Queue deals with memory for numpy arrays
# it seems that the Queue will have a view of the numpy array. This should
# make the Queue fast enough for caching numpy arrays.
import Queue
import numpy

# create a large array... make sure the machine has enough ram for this operation
data = numpy.ones( (100000,100000), dtype='float16')

queue = Queue.Queue(10)

queue.put(data)

# check the ram usage here. Should not have doubled after put'ing data into the
# queue.

# it should be safe to delete data and still have the queue's handle to the
# numpy array
# del data