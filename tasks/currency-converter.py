import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras as ks
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam

in_train = np.array([1, 10, 12, 14, 15, 21, 2.4, 2.6, 76.7, 99, 66.6, 666])
out_train = np.array(
    [
        2.555,
        25.55,
        30.66,
        35.77,
        38.325,
        53.655,
        6.132,
        6.643,
        195.9685,
        252.945,
        170.163,
        1701.63,
    ]
)

model = ks.Sequential()
model.add(Dense(1, input_shape=[1], activation="linear"))
model.compile(loss="mean_squared_error", optimizer=Adam(0.1))

history = model.fit(in_train, out_train, epochs=5000, verbose=False)
print("Обучение завершено!")

print(model.predict([23]))
