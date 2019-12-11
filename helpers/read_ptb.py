"""Load ptb dataset."""
# based off of reader.py from ptb tensorflow example
import os
import collections
import numpy


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(list(counter.items()),
                         key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(list(zip(words, list(range(len(words))))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)

    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None):
    """Load ptb data."""
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
    Yields:
        Pairs of the batched data, each a matrix of shape
        [batch_size, num_steps]. The second element of the tuple is the same
        data time-shifted to the right by one.
    Raises:
        ValueError: if batch_size or num_steps are too high.
    """
    raw_data = numpy.array(raw_data, dtype=numpy.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = numpy.zeros([batch_size, batch_len], dtype=numpy.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield(x, y)
