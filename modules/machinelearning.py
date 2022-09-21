import numpy as np
import matplotlib.pyplot as plt
import random as rd


class Division_By_Two_Category:

    x_train = None
    y_train = None
    w = [0, -1]
    n_train = 0
    N = 50
    L = 0.1
    e = 0.1

    # def a(self, x):
    #     return self.func(x, self.w)

    a = lambda self, x: np.sign(x[0] * self.w[0] + x[1] * self.w[1])

    last_error_index = (
        -1
    )  # индекс последнего ошибочного наблюдения(для работы с переменной e)

    @classmethod
    def help():
        print(
            """структура нейросети(net):
        net = {
            # ! означает обязательное значение
            # [...] означает массив одинаковых типов данных
            # ... означает числовое значение
            # func() означает lambda функцию
            "x" : [...], # !начальные данные
            "y" : [...], # !ответы
            "len" : ..., # размер обучающей выборки(размер данных для обучения)
            "w0" : ..., # начальное значение вектора w(w - весы)(какие-то базовые значения)
            "N" : ..., # максимальное число итераций(число проходов в обучении)
            "L" : ...,  # шаг изменения веса(шаг изменения для весов(веса))
            "e" : ..., # небольшая добавка для w0 чтобы был зазор между разделяющей линией и граничным образом(не обязательное значение)
            "a" : func(), # решающее правило(правило для получение ответа(y))
        }
        """
        )

    def __init__(self, net):
        """структура нейросети(net):
        net = {
            # ! означает обязательное значение
            # [...] означает массив одинаковых типов данных
            # ... означает числовое значение
            # func() означает lambda функцию
            "x" : [...], # !начальные данные
            "y" : [...], # !ответы
            "len" : ..., # размер обучающей выборки(размер данных для обучения)
            "w0" : ..., # начальное значение вектора w(w - весы)(какие-то базовые значения)
            "N" : ..., # максимальное число итераций(число проходов в обучении)
            "L" : ...,  # шаг изменения веса(шаг изменения для весов(веса))
            "e" : ..., # небольшая добавка для w0 чтобы был зазор между разделяющей линией и граничным образом(не обязательное значение)
            "a" : func(), # решающее правило(правило для получение ответа(y))
        }
        """
        self.x_train = np.array(net["x"])
        self.y_train = np.array(net["y"])
        if "len" in net:
            self.n_train = net["len"]
        else:
            self.n_train = len(self.x_train)
        if "w0" in net:
            self.w[0] = net["w0"]
        if "N" in net:
            self.N = net["N"]
        if "L" in net:
            self.L = net["L"]
        if "e" in net:
            self.e = net["e"]
        if "a" in net:
            self.a = net["a"]
        pass

    def learning(self):
        for n in range(self.N):
            for i in range(self.n_train):  # перебор по наблюдениям(обучающим данным)
                if (
                    self.y_train[i] * self.a(self.x_train[i]) < 0
                ):  # если ошибка классификации(тоесть сравнение значения ответа(y_train) и ответа по правилу нейросети(a))(в данном случае происходит проверка на отрицательность\положительность обоих значений, тк у нас только два возможных ответа [-1, 1]),
                    self.w[0] += (
                        self.L * self.y_train[i]
                    )  # то корректировка веса w0(w1 не принимает участие в этом тк она имеет костантное значение(объясняется математически))
                    self.last_error_index = i  # индекс последней ошибки классификации

            Q = sum(
                [
                    1
                    for i in range(self.n_train)
                    if self.y_train[i] * self.a(self.x_train[i]) < 0
                ]
            )  # расчёт качества нашей нейросети(проверка всех вариантов на ошибочность(сумма должна быть равна нулю и тогда ошибок не возникло))
            if Q == 0:  # показатель качества классификации (число ошибок)
                break  # остановка, если все верно классифицируем

        if (
            self.last_error_index > -1
        ):  # если имела место быть ошибка классификации(очень близко к всегда)
            self.w[0] = (
                self.w[0] + self.e * self.y_train[self.last_error_index]
            )  # производим смещение для предотвращения наложения линии на наши начальные данные

    def show_graphic(self):
        """красный == 1
        синий == -1"""
        print(self.w)  # вывод наших весов

        line_x = list(
            range(max(self.x_train[:, 0]))
        )  # формирование графика разделяющей линии(x_train[:, 0] это принцип записи(выборки) у массивов в numpy что означает для двумерного массива взять все данные в первом измерении двумерного массива и взять 0-е данные во втором измерении)
        line_y = [self.w[0] * x for x in line_x]

        x_0 = self.x_train[self.y_train == 1]  # формирование точек для 1-го
        x_1 = self.x_train[self.y_train == -1]  # и 2-го классов

        plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
        plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
        plt.plot(line_x, line_y, color="green")

        plt.xlim([0, 45])
        plt.ylim([0, 75])
        plt.ylabel("длина")
        plt.xlabel("ширина")
        plt.grid(True)
        plt.show()

    def check_result(self, data):
        """data массив как x_train или x"""

        result = []
        for arr in data:
            result.append(self.a(arr))

        return result


# x_train = np.array(
#     [
#         [10, 50],
#         [20, 30],
#         [25, 30],
#         [20, 60],
#         [15, 70],
#         [40, 40],
#         [30, 45],
#         [20, 45],
#         [40, 30],
#         [7, 35],
#     ]
# )  # начальные данные
# y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])  # ответы

# n_train = len(x_train)  # размер обучающей выборки(размер данных для обучения)
# w = [0, -1]  # начальное значение вектора w(w - весы)(какие-то базовые значения)
# a = lambda x: np.sign(
#     x[0] * w[0] + x[1] * w[1]
# )  # решающее правило(правило для получение ответа(y))
# N = 50  # максимальное число итераций(число проходов в обучении)
# L = 0.1  # шаг изменения веса(шаг изменения для весов(веса))
# e = 0.1  # небольшая добавка для w0 чтобы был зазор между разделяющей линией и граничным образом(не обязательное значение)

# last_error_index = (
#     -1
# )  # индекс последнего ошибочного наблюдения(для работы с переменной e)

# for n in range(N):
#     for i in range(n_train):  # перебор по наблюдениям(обучающим данным)
#         if (
#             y_train[i] * a(x_train[i]) < 0
#         ):  # если ошибка классификации(тоесть сравнение значения ответа(y_train) и ответа по правилу нейросети(a))(в данном случае происходит проверка на отрицательность\положительность обоих значений, тк у нас только два возможных ответа [-1, 1]),
#             w[0] = (
#                 w[0] + L * y_train[i]
#             )  # то корректировка веса w0(w1 не принимает участие в этом тк она имеет костантное значение(объясняется математически))
#             last_error_index = i  # индекс последней ошибки классификации

#     Q = sum(
#         [1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0]
#     )  # расчёт качества нашей нейросети(проверка всех вариантов на ошибочность(сумма должна быть равна нулю и тогда ошибок не возникло))
#     if Q == 0:  # показатель качества классификации (число ошибок)
#         break  # остановка, если все верно классифицируем

# if (
#     last_error_index > -1
# ):  # если имела место быть ошибка классификации(очень близко к всегда)
#     w[0] = (
#         w[0] + e * y_train[last_error_index]
#     )  # производим смещение для предотвращения наложения линии на наши начальные данные

# print(w)  # вывод наших весов

# line_x = list(
#     range(max(x_train[:, 0]))
# )  # формирование графика разделяющей линии(x_train[:, 0] это принцип записи(выборки) у массивов в numpy что означает для двумерного массива взять все данные в первом измерении двумерного массива и взять 0-е данные во втором измерении)
# line_y = [w[0] * x for x in line_x]

# x_0 = x_train[y_train == 1]  # формирование точек для 1-го
# x_1 = x_train[y_train == -1]  # и 2-го классов

# plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
# plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
# plt.plot(line_x, line_y, color="green")

# plt.xlim([0, 45])
# plt.ylim([0, 75])
# plt.ylabel("длина")
# plt.xlabel("ширина")
# plt.grid(True)
# plt.show()


class Division_By_Three_Category:
    x_train = None
    y_train = None
    w = None

    @classmethod
    def help():
        print(
            """математическое решение класса Division_By_Two_Category
        структура нейросети(net):
        net = {
            # ! означает обязательное значение
            # [...] означает массив одинаковых типов данных
            "x" : [...], # !начальные данные
            "y" : [...], # !ответы
        }"""
        )

    def __init__(self, net):
        """математическое решение класса Division_By_Two_Category
        структура нейросети(net):
        net = {
            # ! означает обязательное значение
            # [...] означает массив одинаковых типов данных
            "x" : [...], # !начальные данные
            "y" : [...], # !ответы
        }
        """
        self.x_train = [x + [1] for x in net["x"]]
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(net["y"])
        pass

    def learning(self):
        pt = np.sum([x * y for x, y in zip(self.x_train, self.y_train)], axis=0)
        xxt = np.sum([np.outer(x, x) for x in self.x_train], axis=0)
        self.w = np.dot(pt, np.linalg.inv(xxt))

    def show_graphic(self):
        """красный == 1
        синий == -1"""
        print(self.w)  # вывод наших весов

        line_x = list(
            range(max(self.x_train[:, 0]))
        )  # формирование графика разделяющей линии

        line_y = [-x * self.w[0] / self.w[1] - self.w[2] / self.w[1] for x in line_x]

        x_0 = self.x_train[self.y_train == 1]  # формирование точек для 1-го
        x_1 = self.x_train[self.y_train == -1]  # и 2-го классов

        plt.scatter(x_0[:, 0], x_0[:, 1], color="red")
        plt.scatter(x_1[:, 0], x_1[:, 1], color="blue")
        plt.plot(line_x, line_y, color="green")

        plt.xlim([0, 45])
        plt.ylim([0, 75])
        plt.ylabel("длина")
        plt.xlabel("ширина")
        plt.grid(True)
        plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
# x_train = [x + [1] for x in x_train]
# x_train = np.array(x_train)
# y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

# pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)
# xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)
# w = np.dot(pt, np.linalg.inv(xxt))
# print(w)

# line_x = list(range(max(x_train[:, 0])))    # формирование графика разделяющей линии
# line_y = [-x*w[0]/w[1] - w[2]/w[1] for x in line_x]

# x_0 = x_train[y_train == 1]                 # формирование точек для 1-го
# x_1 = x_train[y_train == -1]                # и 2-го классов

# plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
# plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
# plt.plot(line_x, line_y, color='green')

# plt.xlim([0, 45])
# plt.ylim([0, 75])
# plt.ylabel("длина")
# plt.xlabel("ширина")
# plt.grid(True)
# plt.show()
