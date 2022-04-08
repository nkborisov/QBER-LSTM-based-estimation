import numpy as np


class ExponentionalAverager:
    def __init__(self, start_value, window_size):
        assert window_size >= 0
        self.val = start_value
        self.alpha = 2. / (window_size + 1)
        self.window_size = window_size

    def add_value(self, val):
        self.val = self.alpha * val + (1. - self.alpha) * self.val

    def get_value(self):
        return self.val


class VarianceEstimator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.vals = list()

    def add_value(self, val):
        if len(self.vals) > self.window_size:
            self.vals.clear()

        self.vals.append(val)

    def retrieve_sigma(self):
        return np.std(self.vals) if len(self.vals) else 0.


class VarianceEstimator2:
    def __init__(self, window_size):
        self.window_size = window_size
        self.vals = list()

    def add_value(self, val):
        if len(self.vals) > self.window_size:
            self.vals.clear()

        self.vals.append(val)

    def retrieve_sigma(self):
        if not len(self.vals):
            return 0.
        avg = np.mean(self.vals)
        sigma = 0.
        for v in self.vals:
            sigma += (v - avg) * (v - avg)
        sigma /= len(self.vals)
        sigma = np.sqrt(sigma)
        sigma2 = np.std(self.vals) if len(self.vals) else 0.
        return sigma
