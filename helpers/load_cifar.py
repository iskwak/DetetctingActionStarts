"""Load cifar data.

Taken from https://github.com/benanne/theano-tutorial
"""
import numpy as np
import os
# import cPickle as pickle
# import glob
# import cv2
import numpy


def one_hot(x, n):
    """Convert index representation to one-hot representation."""
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch_cifar10(filename, data_dir_cifar10, dtype='float64'):
    """Load a batch in the CIFAR-10 format."""
    path = os.path.join(data_dir_cifar10, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0  # scale between [0, 1]
    # data = batch['data']  # scaling will be done elsewhere
    # convert labels to one-hot representation
    # labels = one_hot(batch['labels'], n=10)
    labels = numpy.asarray(batch['labels'], dtype='int32')
    return data.astype(dtype), labels.astype(dtype)


def _grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def _create_cv_img(img):
    temp = numpy.zeros((32, 32, 3), dtype='float32')
    temp[:, :, 0] = img[2, :, :]
    temp[:, :, 1] = img[1, :, :]
    temp[:, :, 2] = img[0, :, :]

    return temp


def _undo_cv_img(img):
    temp = numpy.zeros((3, 32, 32), dtype='float32')
    temp[0, :, :] = img[:, :, 2]
    temp[1, :, :] = img[:, :, 1]
    temp[2, :, :] = img[:, :, 0]

    return temp


def cifar10(dtype='float64', grayscale=True, data_dir=None):
    """Load cifar10 data."""
    if data_dir is None:
        data_dir = "/localhome/kwaki/data"
    data_dir_cifar10 = os.path.join(data_dir, "cifar-10-batches-py")
    # class_names_cifar10 = np.load(os.path.join(
    #     data_dir_cifar10, "batches.meta"))

    # train
    x_train = []
    t_train = []
    for k in range(5):
        x, t = _load_batch_cifar10("data_batch_%d" % (k + 1), data_dir_cifar10,
                                   dtype=dtype)
        x_train.append(x)
        t_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    t_train = np.concatenate(t_train, axis=0)

    # test
    x_test, t_test = _load_batch_cifar10("test_batch", data_dir_cifar10,
                                         dtype=dtype)

    # convert the labels to int32's (scalars for theano)
    t_train = t_train.astype('int32')
    t_test = t_test.astype('int32')

    # reshape the data
    x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))

    means = []
    stds = []
    for i in range(3):
        means.append(x_train[:, i, :, :].mean())
        stds.append(x_train[:, i, :, :].std())
        x_train[:, i, :, :] = x_train[:, i, :, :] - means[i]
        x_test[:, i, :, :] = x_test[:, i, :, :] - means[i]

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    return x_train, t_train, x_test, t_test


def _load_batch_cifar100(filename, data_dir_cifar100, dtype='float64'):
    """Load a batch in the CIFAR-100 format."""
    path = os.path.join(data_dir_cifar100, filename)
    batch = np.load(path)
    data = batch['data'] / 255.0
    # data = batch['data']
    labels = one_hot(batch['fine_labels'], n=100)
    return data.astype(dtype), labels.astype(dtype)


def cifar100(dtype='float64', grayscale=True, data_dir=None):
    """Load all cifar 100 data."""
    if data_dir is None:
        data_dir = "/localhome/kwaki/data"
    data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")
    # class_names_cifar100 = np.load(os.path.join(data_dir_cifar100, "meta"))

    x_train, t_train = _load_batch_cifar100(
        "train", data_dir_cifar100, dtype=dtype)
    x_test, t_test = _load_batch_cifar100(
        "test", data_dir_cifar100, dtype=dtype)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)

    return x_train, t_train, x_test, t_test
