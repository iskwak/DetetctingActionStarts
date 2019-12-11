"""Helper class for computing running stats."""
import numpy


class RunningStats:
    def __init__(self, dims):
        self.dims = dims
        self.mean = numpy.zeros((dims))
        self.mean2 = numpy.zeros((dims))
        self.var = numpy.ones((dims))
        self.count = 0

    def add_data(self, data):
        num_rows = data.shape[0]
        for i in range(num_rows):
            self.count += 1
            delta = data[i, :] - self.mean
            self.mean += delta / self.count
            delta2 = data[i, :] - self.mean
            self.mean2 += delta * delta2
            if self.count < 2:
                self.var = float("nan")
            else:
                self.var = self.mean2 / (self.count - 1)

    def compute_std(self):
        eps = numpy.finfo("float32").eps
        self.std = numpy.sqrt(self.var) + eps

        return self.std

    def compute_std_noeps(self):
        std = numpy.sqrt(self.var)

        return std

