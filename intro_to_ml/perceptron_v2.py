import numpy as np
from nnt import *

class Perceptron(NeuralNet):

    def __init__(self, eta=0.01, n_iter=10):
        NeuralNet.__init__(self, eta, n_iter)


    def fit(self, X, y):

        if self.w_.shape[0] < (1 + X.shape[1]):
            self.w_ = np.zeros(1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

