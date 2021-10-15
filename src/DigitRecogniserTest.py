import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

model_path ='model/model.h5'
model = load_model(model_path)

img = cv2.imread('numbers/savedImage2.jpg')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
#
gray = ~gray
plt.imshow(gray)
resized = cv2.resize(gray, (28, 28), interpolation =cv2.INTER_AREA)
plt.imshow(resized)
newimg = tf.keras.utils.normalize(resized, axis = 1)
plt.imshow(newimg)


newimg = np.array(newimg).reshape(-1, 28, 28, 1)
predictions = model.predict(newimg)
print(predictions)
print(np.argmax(predictions))