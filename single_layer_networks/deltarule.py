import sys
import numpy as np

sys.path.append(r"C:\Users\Zhenia\PycharmProjects\Machine Learning")
import actf


class Neuron:
    """
    !Для сигмоиды классами должны быть 0 и 1. Для гипер. тангенца -1 и 1. И т. д.
    Элемент с неограниченным числом входов и одним выходом.
    Обучается по методу градиентного спуска.
    Считает что обхекты представлены строками, а веса столбцами.
    Смещением считает последнюю строку в векторе весов.
    """

    def __init__(self, num_inputs,  activation_function=actf.sigmoid, activation_function_derivative=actf.sigmoid_prime):
        """

        :param num_inputs: Число входных признаков (смещение учитывать не нужно).
        :param activation_function: Функция активации.
        :param activation_function_derivative: Производная функции активации по сумматорной функции
        """
        self._init_w(num_inputs)
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative # производная активационной функции

    def _init_w(self, num_inputs):
        """ Инициальзация W, T - последний элемент
        """
        self._w =np.random.rand(num_inputs + 1, 1)
        self._w[num_inputs] = 2 * self._w[num_inputs] - 1

    def netinput(self, X):
        """ Рассчитывает чистый вход, объект - строка признаков
        """
        return np.dot(X, self._w)

    def fit(self, X, y, num_epochs=100, bunch_size=10, eta=0.1, eps=0.001):
        """
        :param X:
        :param y:
        :param num_epochs:
        :param bunch_size:
        :param eta: размер шага
        :return: (количество пройденных эпох, конечную ошибку)
        """
        X = np.concatenate((X, np.ones((X.shape[0], 1), dtype=X.dtype)), axis=1)
        y = y.reshape(y.shape[0], 1)
        obj_numbers = np.arange(y.shape[0])
        for epoch in range(num_epochs):
            np.random.shuffle(obj_numbers)
            for i in np.arange(np.ceil(len(y) / bunch_size)):
                bunch_num = int(i * bunch_size)
                bunch = X[obj_numbers[bunch_num : bunch_num + bunch_size]]  # Выбираем пакет объектов
                target_bunch = y[obj_numbers[bunch_num : bunch_num + bunch_size]]   # Их целевые значения
                sum_b = self.netinput(bunch)
                error = 0.5 * np.mean((self.activation_function(self.netinput(X)) - y) ** 2)
                grad = (1/len(y)) * (self.activation_function(sum_b) - target_bunch) \
                       * self.activation_function_derivative(sum_b) * bunch
                grad = grad.sum(axis=0) # суммируем приращение весов от всех объектов
                grad = grad.reshape((grad.shape[0], 1))  # переворачиваем для вычитания из вектора весов
                self._w -= eta * grad
                error_new = 0.5 * np.mean((self.activation_function(self.netinput(X)) - y) ** 2)
                if np.abs(error - error_new) < eps:
                    return epoch + 1, np.abs(error - error_new)
        return epoch + 1, np.abs(error - error_new)

    def predict(self, X):
        try:
            X = np.concatenate((X, np.ones((X.shape[0], 1), dtype=X.dtype)), axis=1)
        except np.AxisError:
            X = np.append(X, 1)
        return self.activation_function(self.netinput(X))


if __name__ == '__main__':
    # simple data example.
    X = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    y = np.array([0, 1, 1, 1])

    n = Neuron(3)
    assert n._w.shape == (4, 1), "Initialization of weights work wrong."

    n = Neuron(2)
    print(n.fit(X, y, bunch_size=10, num_epochs=1000, eps=0.0001, eta=0.1))
    print(n.predict(X))