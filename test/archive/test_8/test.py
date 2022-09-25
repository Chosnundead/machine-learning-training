from distutils.command.upload import upload
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO

PATH_TO_CURRENT_FILE = Path(__file__).parents[0]
model = keras.applications.VGG19()

img = Image.open("{}/image.jpg".format(PATH_TO_CURRENT_FILE))
plt.imshow(img)
plt.show()

# приводим к входному формату VGG-сети
img = np.array(img)
x = keras.applications.vgg19.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)

# прогоняем через сеть
res = model.predict(x)
print(np.argmax(res))
