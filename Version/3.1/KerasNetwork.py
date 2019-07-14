from __future__ import print_function

import tensorflow as tf
from PhotoHandler import get_training_set_and_test_set
from PhotoHandler import get_CNN_training_set_and_test_set
from PhotoHandler import CNN_jpg_width
from PhotoHandler import CNN_jpg_height

import sys

conv_num = int(sys.argv[1])
dp1 = float(sys.argv[2])
dp2 = float(sys.argv[3])
dense_num = int(sys.argv[4])
print("For number of cells in Conv2D=%d" % (conv_num))
print("For Dropout_1=%.1f Dropout_2=%.1f" % (dp1, dp2))
print("For number of cells in Dense=%d" % (dense_num))

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
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(conv_num, (3, 3), input_shape=(CNN_jpg_width, CNN_jpg_height, 3), activation=tf.nn.relu),
    tf.keras.layers.Conv2D(conv_num, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(dp1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(dense_num, activation=tf.nn.relu),
    tf.keras.layers.Dropout(dp2),
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
model.evaluate(x_test, y_test) 
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)