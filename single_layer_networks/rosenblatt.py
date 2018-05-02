import sys
sys.path.append(r"C:\Users\Zhenia\PycharmProjects\Machine Learning")
import numpy as np
import actf


class Perceptron():
    """
    Для простоты буду использовать Перцептрон с одним выходом.
    Смещение включено в self._w: последняя строка.
    """
    def __init__(self, X):
        self._init_w(X)

    def fit(self, X, y, num_iter=3000, alpha = 1, warm_start=False):
        """
        Должен сходиться если данные линейно разделимы.
        :param X: массив (n,m), n - число объектов.
        :param y: массив размером (1,n), состоящий из 0 и 1
        :param num_iter: максимальное число итераций на случай линейно неразделимых данных.
        :param alpha: default 1
        :param warm_start: продолжить с существующими весами?
        :return:
        """
        if not warm_start:
            self._init_w(X)
        y = y.reshape(y.shape[0], 1)
        X = np.concatenate( (X, -1 * np.ones((X.shape[0], 1)) ), axis=1)
        self.errors = []
        for _ in range(num_iter):
            num_errors = 0
            for idy, obj_y in enumerate(y):     # для кажого объекта.
                pred = self.predict_fit(X[idy])
                self._w += (y[idy] - pred)*X[idy].reshape(X[idy].shape[0],1)
                num_errors += 1
            self.errors.append(num_errors)
            if num_errors == 0:
                break

    def train_on_single_example(self, example, y):
        example = np.append(example, -1)
        error = y - self.predict_fit(example)
        self._w += error * example.reshape(example.shape[0],1)
        return error


    def netinput(self, X):
        """
        Рассчитывает чистый вход
        """
        return np.dot(X, self._w)

    def predict_fit(self, X):
        """
        Предсказание с помощью пороговой функции
        """
        return self.netinput(X) > 0

    def predict(self, X):
        """
        Предсказание (для внешнего вызова)
        :param X:
        :return:
        """
        X = np.concatenate((X, -1 * np.ones((X.shape[0], 1))), axis=1)
        return self.netinput(X) > 0

    def _init_w(self, X):
        """ Инициальзация W случайными числами (смещение включено в self._w: последняя строка.)
        """
        self._w = np.random.rand(X.shape[1] + 1, 1)


if __name__ == '__main__':
    X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    y = np.array([-1, 1, 1, 1])
    p = Perceptron(X)
    p.fit(X, y, num_iter=100, alpha=50)
    print(p.predict(X))
    print(p._w[:-1])
    print(p._w[-1:])

