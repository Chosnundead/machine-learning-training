# Импорт библиотек
import keras as ks
import numpy as np
from keras.layers import Dense, Flatten

# Глобальные значения(кол-во входных значений)
AMOUNT_OF_OBS_DATA = 4

# Создание нашего класса для нейросети
class AI:

    # Создать модель нейросети
    def __init__(self):
        self.model = ks.Sequential(
            [
                Flatten(input_shape=(AMOUNT_OF_OBS_DATA,)),
                Dense(2, activation="relu"),
                Dense(1, activation="relu"),
            ]
        )
        # self.model.compile(
        #     optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        # )
        pass

    # Выведение инфы
    def _info(self):
        print(self.model.summary())
        pass

    # Получение весов
    def get_weights(self):
        resultArray = []
        for layer in self.model.layers:
            layerArray = layer.get_weights()
            print(layerArray)
            counter = 0
            for array in layerArray:
                counter += 1
                if counter % 2 == 0:
                    continue
                for number in array.flatten().tolist():
                    resultArray.append(number)
        return resultArray

    # Изменение весов на другие
    def set_weights(self, arr):
        layers = []
        layers.append([])
        layers.append(
            [
                np.array([[arr[i], arr[i + 1]] for i in range(0, 8, 2)]),
                np.array([0.0, 0.0]),
            ]
        )
        layers.append([np.array([[arr[i]] for i in range(8, 10)]), np.array([0.0])])
        counter = 0
        for layer in self.model.layers:
            layer.set_weights(layers[counter])
            counter += 1
        pass
