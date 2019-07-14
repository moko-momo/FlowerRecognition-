from __future__ import print_function

import tensorflow as tf
from PhotoHandler import get_training_set_and_test_set


(x_train, y_train),(x_test, y_test) = get_training_set_and_test_set()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(30000, )),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print("Final Test")
model.evaluate(x_test, y_test)