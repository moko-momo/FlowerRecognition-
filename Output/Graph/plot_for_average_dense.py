import os
import math
import numpy as np
import matplotlib.pyplot as plt

file_names = os.listdir("./../Formatted_Output")
evaluate_acc_list_128 = []
evaluate_acc_list_512 = []

for file_name in file_names:
    if file_name[0] == "F" and (file_name[12] == "f" or file_name[13] == "f"):
        f = open(str("./../Formatted_Output/" + file_name))
        line = f.readline()
        temp_evaluate_acc_list = []
        while line != '':
            line_list = line.split()
            if len(line_list) == 0:
                line = f.readline()
                continue
            if line_list[0] == "Evaluation":
                temp_evaluate_acc_list.append(float(line_list[-1]))
            line = f.readline()
        if file_name[-7:-4] == "128":
            evaluate_acc_list_128.append(temp_evaluate_acc_list)
        else:
            evaluate_acc_list_512.append(temp_evaluate_acc_list)
        f.close()


# calulate average
average_128 = []
average_512 = []
sum = 0
n = 5

for count in range(0, n):
    for i in evaluate_acc_list_128:
        sum += i[count]
    average_128.append(float(sum / 10))
    sum = 0

for count in range(0, n):
    for i in evaluate_acc_list_512:
        sum += i[count]
    average_512.append(float(sum / 10))
    sum = 0

temp_x = [1, 2, 3, 4, 5]
y_128 = []
y_512 = []
x = []

for i in evaluate_acc_list_128:
    for count in range(0, n):
        y_128.append(i[count])

for i in evaluate_acc_list_512:
    for count in range(0, n):
        y_512.append(i[count])

for i in range(0, n * 10):
    x.append(temp_x[i % n])


plt.figure()
plt.plot(temp_x, average_128, "-", label="Dense=128")
plt.plot(temp_x, average_512, "-", label="Dense=512")
plt.plot(x, y_128, "x")
plt.plot(x, y_512, "x")
plt.legend()
plt.savefig("./dense_average.jpg", dpi=200)