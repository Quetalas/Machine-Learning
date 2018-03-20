import numpy as np
from nnt import *


class AdalineGd(NeuralNet):
    """ Классификатор на основе ADALINE (ADAptive Linear Neuron).
    """

    def __init__(self, eta=0.01, n_iter=50):
        NeuralNet.__init__(self, eta, n_iter)

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        """ Рассчитать линейную активацию"""
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1,-1)