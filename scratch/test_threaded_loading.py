# import Queue
# import threading
import numpy as np

import test_threaded_loading_helper
reload(test_threaded_loading_helper)


def sampler(opts, data, labels):
    """Sampler."""
    num_elements = opts["num_elements"]
    minibatch_size = opts["minibatch_size"]
    idxs = np.random.choice(num_elements, minibatch_size, replace=False)
    data = data[idxs]
    labels = labels[idxs]

    return [data, labels]


temp = np.transpose(np.asarray([range(10)]))
data = np.tile(temp, (1, 4))
labels = np.asarray(range(10))

opts = {
    "num_elements": data.shape[0],
    "minibatch_size": 3
}

loader = test_threaded_loading_helper.DataLoaderPool(
    sampler=sampler, params=[opts, data, labels])

for i in range(10):
    (examples, labels) = loader.get_minibatch()
    print (examples, labels)

del loader
