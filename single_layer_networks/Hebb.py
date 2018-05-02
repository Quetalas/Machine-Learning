import sys
sys.path.append(r"C:\Users\Zhenia\PycharmProjects\Machine Learning")
import numpy as np
import actf


class PerHebba():
    """ Обучение по правилу Хебба
    """
    def fit(self, X, y):
        T = np.full((X.shape[0], 1), -1)
        X = np.append(X, T, axis=1)
        self._w = np.array([X.T.dot(y)]) # [[2,2,-2]] - веса для первого нейрона в сети из одного нейрона

    def trace_fit(self,X, y):
        T = np.full((X.shape[0], 1), -1)
        X = np.append(X, T, axis=1)
        self._w = np.zeros((1,X.shape[1]))
        for x, yi in zip(X, y):
            for xi, j in zip(x, range(self._w.shape[1])):
                self._w[0, j] += xi*yi
            yield self._w

    def predict(self, X):
        T = np.full((X.shape[0], 1), -1)
        X = np.append(X, T, axis=1)
        S = X.dot(self._w.T)
        pred = []
        for y in S.flat:
            pred.append(actf.reLu(y))
        return np.array(pred)