from __future__ import print_function

import tensorflow as tf
from PhotoHandler import get_training_set_and_test_set
from PhotoHandler import get_CNN_training_set_and_test_set
from PhotoHandler import CNN_jpg_width
from PhotoHandler import CNN_jpg_height

import sys

dp1 = float(sys.argv[1])
dp2 = float(sys.argv[2])
print("For Dropout_1=%.1f Dropout_2=%.1f" % (dp1, dp2))

# (x_train, y_train),(x_test, y_test) = get_training_set_and_test_set()
(x_train, y_train),(x_test, y_test) = get_CNN_training_set_and_test_set()
x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(30000, )),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(5, activation=tf.nn.softmax)
# ])

# CNN Model
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Conv2D(128, (3, 3), input_shape=(CNN_jpg_width, CNN_jpg_height, 3), activation=tf.nn.relu),
#    tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
#    tf.keras.layers.MaxPool2D(),
#    tf.keras.layers.Dropout(dp1),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(512, activation=tf.nn.relu),
#    tf.keras.layers.Dropout(dp2),
#    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
#])
#
# AlexNet Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), input_shape=(CNN_jpg_width, CNN_jpg_height, 3), activation=tf.nn.relu, padding='valid', kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer='uniform'),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer='uniform'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu, kernel_initializer='uniform'),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test) # ------------------------ BEST: dp1=0.6, dp2=0.6 and acc=0.6081 loss=1.0303
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)