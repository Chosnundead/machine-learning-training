import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import keras as ks
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.optimizers import Adam
from pathlib import Path

PATH_TO_CURRENT_FILE = Path(__file__).parents[0]

model = None
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = ks.utils.to_categorical(y_train, 10)
y_test = ks.utils.to_categorical(y_test, 10)


def menu():
    global PATH_TO_CURRENT_FILE, model
    mapOfInputs = {0: "Выход", 1: "Загрузить нейронку", 2: "Создать нейронку"}

    if model != None:
        mapOfInputs[3] = "Протестировать"
        mapOfInputs[4] = "Сохранить нейронку"

    print("=====================")
    for key in mapOfInputs:
        print("{}:\t{}".format(key, mapOfInputs.get(key)))
    print("Введите ваш выбор: ", end="")
    select = int(input())
    print("=====================")

    if select in mapOfInputs:
        if select == 0:
            return True
        elif select == 1:
            model = ks.models.load_model(
                "{}/saved-ai/image-classification".format(PATH_TO_CURRENT_FILE)
            )
        elif select == 2:
            model = create_model()
        elif select == 3:
            test_model(model)
        elif select == 4:
            model.save("{}/saved-ai/image-classification".format(PATH_TO_CURRENT_FILE))

    return False


def create_model():
    global x_train, x_test, y_train, y_test

    model = ks.Sequential(
        [
            Flatten(input_shape=(32, 32, 3)),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(66, activation="relu"),
            Dense(21, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    print(model.summary())

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=21, epochs=10, validation_split=0.2)

    model.evaluate(x_test, y_test)

    return model


def get_title_of_image(number):
    number = int(number)
    if number == 0:
        return "airplane"
    elif number == 1:
        return "automobile"
    elif number == 2:
        return "bird"
    elif number == 3:
        return "cat"
    elif number == 4:
        return "deer"
    elif number == 5:
        return "dog"
    elif number == 6:
        return "frog"
    elif number == 7:
        return "horse"
    elif number == 8:
        return "ship"
    elif number == 9:
        return "truck"
    else:
        print("Error!")
        raise Exception


def test_model(model):
    global x_train, x_test, y_train, y_test

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    argmax_y_test = np.argmax(y_test, axis=1)
    print(argmax_y_test)

    mask = pred == argmax_y_test
    x_false = x_test[~mask]
    y_false_real = argmax_y_test[~mask]
    y_false_ai = pred[~mask]
    x_true = x_test[mask]
    y_true_real = argmax_y_test[mask]
    y_true_ai = pred[mask]

    for i in range(min(len(x_true), 9)):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow((x_true[i] * 255).astype(int))
        plt.title(f"ai: {get_title_of_image(y_true_ai[i])}")
    plt.show()
    for i in range(min(len(x_false), 9)):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow((x_false[i] * 255).astype(int))
        plt.title(
            f"ai: {get_title_of_image(y_false_ai[i])}; but: {get_title_of_image(y_false_real[i])}"
        )
    plt.show()


# n = 1
# x = np.expand_dims(x_test[n], axis=0)
# res = model.predict(x)
# res = np.argmax(res)
# res = get_title_of_image(res)
# true_res = get_title_of_image(np.argmax(y_test[n]))
# print(f"On this image {true_res}, and our AI said that this is {res}")
# plt.imshow((x_test[n] * 255).astype(int))
# plt.show()


while True:
    if menu():
        break
