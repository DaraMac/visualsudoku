import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, Dropout, Flatten, MaxPooling2D

image_size = 28
# Loading the MNIST data set with samples and splitting it
mnist = tf.keras.datasets.mnist
(x_training_data,y_training_labels), (x_test_data,y_test_labels) = mnist.load_data()
print("number of training datasets & size: ", x_training_data.shape)
print("number of test datasets & size: ", x_test_data.shape)
plt.matshow(x_training_data[10],cmap = plt.cm.binary)


# Normalizing the data (so length = 1)
x_training_data = tf.keras.utils.normalize(x_training_data, axis=1) 
x_test_data = tf.keras.utils.normalize(x_test_data, axis=1)
plt.matshow(x_training_data[10],cmap = plt.cm.binary)

# Add on more dimension for kernel operation
x_training_data = np.array(x_training_data).reshape(-1, image_size, image_size, 1)
x_test_data = np.array(x_test_data).reshape(-1, image_size, image_size, 1)


model = Sequential()
#default model
# =============================================================================
# model.add(Conv2D(64, kernel_size=(3,3), input_shape= x_training_data.shape[1:]))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(64, kernel_size=(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Conv2D(64, kernel_size=(3,3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation("relu"))
# 
# model.add(Dense(32))
# model.add(Activation("relu"))
# 
# model.add(Dense(10))
# model.add(Activation("softmax"))
# 
# =============================================================================
#------------------------------------------
#model 6
# =============================================================================
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# =============================================================================

#model 7
# =============================================================================
# model.add(Conv2D(32, kernel_size=(3,3),activation='relu', input_shape=(28,28,1)))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10,activation='softmax'))
# =============================================================================
   
          
          
#model 8         
# =============================================================================
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))         
# =============================================================================
          
          
#model 9    
# =============================================================================
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2)))    
# model.add(Flatten())
# # Densely connected layers
# model.add(Dense(128, activation='relu'))
# # output layer
# model.add(Dense(10, activation='softmax'))
# 
# =============================================================================

#model 10

# =============================================================================
# model.add(Conv2D(20, (5, 5),
#                  padding = "same", 
#                  input_shape = (28, 28, 1)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
# 
# model.add(Conv2D(50, (5, 5),
#                  padding = "same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
# 
# # Fully connected layers (w/ RELU)
# model.add(Flatten())
# model.add(Dense(500))
# model.add(Activation("relu"))
# 
# # Softmax (for classification)
# model.add(Dense(10))
# model.add(Activation("softmax"))
# =============================================================================

          
model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_training_data, y_training_labels, epochs=10 , validation_split = 0.3)

model.evaluate(x_test_data, y_test_labels)

model.save('model10.h5')





















predictions = model.predict(x_test_data)
y_predicted_labels = [np.argmax(i) for i in predictions]
plt.matshow(x_test_data[266])
predictions[266]
print("predrction is:", np.argmax(predictions[266]))

cm = tf.math.confusion_matrix(labels = y_test_labels, predictions = y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt ='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


np.set_printoptions(suppress=True)
print(y_test_labels[0])
print(predictions[0])
