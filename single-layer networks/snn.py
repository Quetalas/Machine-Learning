import sys
sys.path.append(r"C:\Users\Zhenia\PycharmProjects\Machine Learning")
import numpy as np
import actf

class Perceptron:

    def __init__(self, init_w=True, r=1, c=1):
        self._ini_w(r=r, c=c)

    def _ini_w(self, init=True, r=1, c=1):
        if init:
            self._w = np.random.rand(r, c + 1) # +1 for T

class PerHebba(Perceptron):
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



if __name__ == '__main__':
    X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    y = np.array([-1, 1, 1, 1])
    p = PerHebba()
    p.fit(X, y)
    print(p._w)
    print(p.predict(X))