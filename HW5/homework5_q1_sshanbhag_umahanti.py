# -*- coding: utf-8 -*-
"""homework5_Q1_sshanbhag_umahanti.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L00syQxC8UHI4XSSkpbVsGrJSa-4hFmU
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

fashion_mnist = tf.keras.datasets.fashion_mnist

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train.shape:", x_train.shape, "y_train.shape:", y_train.shape)

#normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#Breaking the training data into training and validation sets 
(x_train, x_valid) = x_train[6000:], x_train[:6000]
(y_train, y_valid) = y_train[6000:], y_train[:6000]

# Reshape input data from (28, 28) to (28, 28, 1)
l, w = 28, 28

x_train = x_train.reshape(x_train.shape[0], l, w, 1)
x_test = x_test.reshape(x_test.shape[0], l, w, 1)
x_valid = x_valid.reshape(x_valid.shape[0], l, w, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#printing the training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

#printing the number of training, validation, and test datasets

print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')
print(x_valid.shape[0], 'validation set')



model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', strides=1, activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

x = model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2, validation_data=(x_valid, y_valid), callbacks=[checkpointer])

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])