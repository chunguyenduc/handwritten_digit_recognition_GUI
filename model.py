import tensorflow as tf
import numpy as np
from tensorflow import keras


mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# One hot coding label
y_train = keras.utils.to_categorical(y_train).astype('int32')


# Our model, 2 hidden layer each has 128 nodes
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)


model.save("model.h5")

#y_pred = model.predict_classes(X_test)
#con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
