import os

# Строка для сокрытия строки с предупреждением
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Импорт библиотек
import random
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from deap import base, algorithms, creator, tools
from neural_net import AI

# Создаем окружение
global env
env = gym.make("CartPole-v1")
obs, info = env.reset()
env.render()
# obs, reward, isDone, isTruncated, info = env.step(0)


# Создаем нейронку
global neuralNet
neuralNet = AI()
neuralNet.get_weights()


# Глобальные значения для обучения
LEN_OF_GEN = len(neuralNet.get_weights())
POPULATION_LEN = 10
MAX_GENERATIONS = 20
MAX_MOVEMENTS = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
LOW = -2.0
UP = 2.0
ETA = 20
HALL_OF_FAME_SIZE = 2

# Создание зала славы поколений
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Объявление базовых классов для пакета deap
# Класс для максимизации награждения
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Класс нашего индивида, что наследуется от класса list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Создание базовых функций для пакета deap и их регистрация в нашем тулбоксе
toolbox = base.Toolbox()
# Функция создания случайного значения в пределах наших LOW и UP для весов
toolbox.register("randomWeight", random.uniform, LOW, UP)
# Функция создания индивида
toolbox.register(
    "individualCreator",
    tools.initRepeat,
    creator.Individual,
    toolbox.randomWeight,
    LEN_OF_GEN,
)
# Функция создания популяции
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# Создание популяции
population = toolbox.populationCreator(n=POPULATION_LEN)

# Объявление функции для расчёта награждения у нашего индивида
def getScore(ind):
    global neuralNet, env

    obs, info = env.reset()
    neuralNet.set_weights(ind)

    resultReward = 0
    counter = 0
    isDone = False
    while not isDone and counter <= MAX_MOVEMENTS:
        arr = np.expand_dims(obs, axis=0)
        pred = neuralNet.model.predict(arr)
        pred = pred.flatten()[0]
        pred = int(min(round(pred), 1))
        print(pred)
        obs, reward, isDone, isTruncated, info = env.step(int(pred))
        resultReward += reward
        counter += 1

    return (resultReward,)


# Создание базовых функций для пакета deap и их регистрация в нашем тулбоксе
# Функция для возврашения score нашего индивида
toolbox.register("evaluate", getScore)
# Функция для выборки индивидов из поколений
toolbox.register("select", tools.selTournament, tournsize=3)
# Функция для наследственности(кроссовера) из двух роителей в новое поколение
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
# Функция для мутирования особей
toolbox.register(
    "mutate",
    tools.mutPolynomialBounded,
    low=LOW,
    up=UP,
    eta=ETA,
    indpb=1.0 / LEN_OF_GEN,
)

# Создание статистики поколений
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

# Запуск нашего обучения
population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    halloffame=hof,
    stats=stats,
    verbose=True,
)

# Выборка значений нашей статистики
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")


# Отрисовка графика
plt.plot(maxFitnessValues, color="red")
plt.plot(meanFitnessValues, color="green")
plt.xlabel("Поколение")
plt.ylabel("Макс(красный)/средняя(зелёная) приспособленность")
plt.title("Зависимость максимальной и средней приспособленности от поколения")
plt.show()

# Получение и отрисовка лучшего индивида из зала славы поколений
best = hof.items[0]
print(f"Лучший парень: {best}.")


# Показ работы лучшего индивида с текстовом варианте
obs, info = env.reset()
neuralNet.set_weights(best)
counter = 0
isDone = False
while not isDone and counter <= MAX_MOVEMENTS:
    arr = np.expand_dims(obs, axis=0)
    pred = neuralNet.model.predict(arr)
    pred = pred.flatten()[0]
    pred = int(min(round(pred), 1))
    print(pred)
    obs, reward, isDone, isTruncated, info = env.step(int(pred))
    counter += 1
print(f"Он выжил на протяжении: {counter} ходов.")

#####################
# import modules.machinelearning as ml
# task1 = ml.Division_By_Two_Category(
#     {
#         "x": [
#             [10, 50],
#             [20, 30],
#             [25, 30],
#             [20, 60],
#             [15, 70],
#             [40, 40],
#             [30, 45],
#             [20, 45],
#             [40, 30],
#             [7, 35],
#         ],
#         "y": [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
#     }
# )
# task1.learning()
# task1.show_graphic()
# print(task1.check_result([[10, 40], [40, 1]]))
#####################
# task2 = ml.Division_By_Three_Category(
#     {
#         "x": [
#             [10, 50],
#             [20, 30],
#             [25, 30],
#             [20, 60],
#             [15, 70],
#             [40, 40],
#             [30, 45],
#             [20, 45],
#             [40, 30],
#             [7, 35],
#         ],
#         "y": [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
#     }
# )
# task2.learning()
# task2.show_graphic()
#####################
# task3 = ml.Division_By_Three_Category_SGD(
#     {
#         "x": [
#             [10, 50],
#             [20, 30],
#             [25, 30],
#             [20, 60],
#             [15, 70],
#             [40, 40],
#             [30, 45],
#             [20, 45],
#             [40, 30],
#             [7, 35],
#         ],
#         "y": [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1],
#     }
# )
# task3.learning()
# task3.show_graphic()
# print(task3.check_result([[10, 40], [40, 1]]))
# task3.show_graphic_of_q_plot()
#####################
