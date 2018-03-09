""" Классы для построения простой нейронной сети
"""
import numpy as np

class Perceptron:
    """ Классификатор на основе персептрона.

    Параметры
    ---------
    eta : float
        Темп обученяи (между 0.0 и 1.0)
    n_iter : int
        Число проходов по тренировочному набору данных.

    Атрибуты
    --------
    w_ : 1-мерный массив
        Весовые коэффициенты после подгонки.
    errors_ : список
        Число случаев ошибочной классификации в каждой эпохе.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Выполнить подгонку модели под тренировочные данные.

        Параметры
        ---------
        X : {массивоподобный}, форма = [n_samples, n_features]
            тренировочные векторы, где
            n_samples - число образцов и
            n_features - число признаков
        y : массивоподобный, форма = [n_samples]
            Целевые значения.

        Возвращает
        ----------
        self : object
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ Рассчитать чистый вход
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ Вернуть метку класса после еденичного скачка
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)