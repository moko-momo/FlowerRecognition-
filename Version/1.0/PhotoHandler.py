from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
from PIL import Image

data_top_path = "./flowers/"
flower_labels = os.listdir(data_top_path)
label_size = len(flower_labels)
jpg_width = 100
jpg_height = 100

def read_jpg_files(file_path):
    origin_jpg_file = Image.open(file_path)
    resized_jpg_file = origin_jpg_file.resize((jpg_width, jpg_height))
    return resized_jpg_file

def read_files_from_fold(fold_path):
    file_list = []
    jpg_name_list = []
    for file_name in os.listdir(fold_path):
        if file_name[-4:] != ".jpg":
            continue
        jpg_file_name = str(fold_path + "/" + file_name)
        jpg_name_list.append(jpg_file_name)
        file_list.append(read_jpg_files(jpg_file_name))
    return (jpg_name_list, file_list)


def get_feature_vector(image):
    ans_list = []
    for i in range(0, jpg_width):
        for j in range(0, jpg_height):
            temp_rgba_info = image.getpixel((i, j))
            for k in range(0, len(temp_rgba_info)):
                ans_list.append(temp_rgba_info[k])
    return np.array(ans_list)

def read_one_class(class_label_number):
    class_name = flower_labels[class_label_number]
    ans_matrix_list = []
    image_file_name_list, image_file_list = read_files_from_fold(str(data_top_path + class_name))
    for i in range(0, len(image_file_name_list)):
        ans_matrix_list.append(get_feature_vector(image_file_list[i]))
    print(str(flower_labels[class_label_number] + " loaded"))
    np_ans_matrix_list = np.array(ans_matrix_list)
    np_labels = np.ones((np_ans_matrix_list.shape[0], 1)) * class_label_number 
    return (np_ans_matrix_list, np_labels)

def load_data():
    test_list = []
    label_list = []
    for i in range(0, len(flower_labels)):
        temp_test, temp_label = read_one_class(i)
        for j in range(0, len(temp_test)):
            test_list.append(temp_test[j])
            label_list.append(temp_label[j])
    return (np.array(test_list), np.array(label_list))

def shuffle(test_list, label_list):
    state = np.random.get_state()
    np.random.shuffle(test_list)
    np.random.set_state(state)
    np.random.shuffle(label_list)
    return (test_list, label_list)

def get_training_set_and_test_set():
    origin_data_set, origin_label_set = load_data()
    data_set, label_set = shuffle(origin_data_set, origin_label_set)
    divide_line = math.floor(4 * data_set.shape[0] / 5)
    x_train = data_set[:divide_line]
    y_train = label_set[:divide_line]
    x_test = data_set[divide_line:]
    y_test = label_set[divide_line:]
    return ((x_train, y_train), (x_test, y_test))
    
    


