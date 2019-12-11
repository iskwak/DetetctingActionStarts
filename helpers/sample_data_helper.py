import numpy
import theano


def create_centers(rng, num_classes, dims):
    # randomly create centers for the classes
    centers = numpy.asarray(
        rng.uniform(low=-5.0, high=5.0, size=(num_classes, dims)),
        dtype=theano.config.floatX
    )
    return centers


def create_sample_data(rng, centers, num_classes, num_samples, dims):
    # create test data to make sure the neural network learns properly
    # fake data
    x_vals = numpy.zeros(
        (num_samples, dims),
        dtype=theano.config.floatX
    )
    y_vals = numpy.ones(
        (num_samples,),
        dtype='int64'
    )

    # training data
    for i in range(num_classes):
        label = i
        start_idx = i * num_samples / num_classes
        end_idx = (i + 1) * num_samples / num_classes
        idx_range = list(range(start_idx, end_idx))
        y_vals[idx_range] = label * y_vals[idx_range]

        x_vals[idx_range] = rng.normal(
            loc=centers[i], size=(num_samples / num_classes, dims))

    return x_vals, y_vals


def create_sample_sequences(rng, centers, num_classes, num_samples, seq_length, dims):
    # create test data to make sure the neural network learns properly
    # fake data
    x_vals = numpy.zeros(
        (num_samples, seq_length, dims),
        dtype=theano.config.floatX
    )
    y_vals = numpy.ones(
        (num_samples, seq_length),
        dtype='int64'
    )

    # training data
    for i in range(num_samples):

        # create a sequence of random numbers
        labels = numpy.random.choice(num_classes,
                                     seq_length,
                                     replace=True)

        for j in range(seq_length):
            y_vals[i, j] = labels[j]

            #x_vals[i, j] = centers[labels[j]]
            x_vals[i, j] = rng.normal(
                loc=centers[labels[j]], size=(dims,))

    return x_vals, y_vals

# # testing data
# x_test = numpy.zeros(
#     (num_test, dim_in),
#     dtype=theano.config.floatX
# )
# y_test = numpy.ones(
#     (num_test,),
#     dtype='int64'
# )

# for i in range(num_classes):
#     label = i
#     start_idx = i * num_test / num_classes
#     end_idx = (i + 1) * num_test / num_classes
#     idx_range = range(start_idx, end_idx)
#     y_test[idx_range] = label * y_test[idx_range]

#     x_test[idx_range] = rng.normal(
#         loc=centers[i], size=(num_test / num_classes, dim_in))
