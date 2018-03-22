import numpy as np


class NeuralNet:

    def __init__(self, eta=0.01, n_iter=50, numfeatures = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.init_weights(numfeatures)

    def init_weights(self, numfeatures):
        """ Пример инициализации весов для одного нейрона
        w_ : [0, 0, 0, 0, ... , 0]
        """
        self.w_ = np.zeros(1 + numfeatures)

    def fit(self, X, y):
        """ Обучение"""
        pass

    def net_input(self, X):
        """ Чистый вход
        Input:
        X : [[объект 1],
            [объект 2],
            .......... ,
            [объект n]]
        Output:
            [[вход объекта 1],
            ............... ,
            [вход объекта n]]
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ Сделать предсказание для обхектов из массива X
        X : [[object 1],
            .... ,
            [object n]]
        Out: [-1,1,1,1, ... , -1]
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)