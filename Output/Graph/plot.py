import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


legend = False
point = True

file_names = os.listdir("./../Formatted_Output")

train_loss_list = []
train_acc_list = []

evaluate_loss_list = []
evaluate_acc_list = []

for file_name in file_names:
    if file_name[0] == "F":
        f = open(str("./../Formatted_Output/" + file_name))
        line = f.readline()
        temp_train_loss_list = []
        temp_train_acc_list = []
        temp_evaluate_loss_list = []
        temp_evaluate_acc_list = []
        while line != '':
            line_list = line.split()
            if len(line_list) == 0:
                line = f.readline()
                continue
            if line_list[0] == "Train":
                temp_train_loss_list.append(float(line_list[-4]))
                temp_train_acc_list.append(float(line_list[-1]))
            if line_list[0] == "Evaluation":
                temp_evaluate_loss_list.append(float(line_list[-4]))
                temp_evaluate_acc_list.append(float(line_list[-1]))
            line = f.readline()
        train_acc_list.append(temp_train_acc_list)
        train_loss_list.append(temp_train_loss_list)
        evaluate_acc_list.append(temp_evaluate_acc_list)
        evaluate_loss_list.append(temp_evaluate_loss_list)
        f.close()
    



x = [1, 2, 3, 4, 5]

count = 0
plt.figure()
plt.rcParams['figure.figsize'] = (8.0, 4.0)
for i in train_acc_list:
    if point == True:
        plt.plot(x, i, 'o', label=str(count))
    else:
        plt.plot(x, i, label=str(count))
    count += 1
plt.title("Train Acc")
if legend == True:
    plt.legend(ncol=5, loc='lower right')
plt.savefig("./train_acc.jpg", dpi=200)


count = 0
plt.figure()
for i in train_loss_list:
    if point == True:
        plt.plot(x, i, 'o', label=str(count))
    else:
        plt.plot(x, i, label=str(count))
    count += 1
plt.title("Train Loss")
if legend == True:
    plt.legend(ncol=5, loc='lower right')
plt.savefig("./train_loss.jpg", dpi=200)


count = 0
plt.figure()
for i in evaluate_acc_list:
    if point == True:
        plt.plot(x, i, 'o', label=str(count))
    else:
        plt.plot(x, i, label=str(count))
    count += 1
plt.title("Evaluation Acc")
if legend == True:
    plt.legend(ncol=5, loc='lower right')
plt.savefig("./evaluate_acc.jpg", dpi=200)


count = 0
plt.figure()
for i in evaluate_loss_list:
    if point == True:
        plt.plot(x, i, 'o', label=str(count))
    else:
        plt.plot(x, i, label=str(count))
    count += 1
plt.title("Evaluation Loss")
if legend == True:
    plt.legend(ncol=5, loc='lower right')
plt.savefig("./evaluate_loss.jpg", dpi=200)



# Draw 3D
parameter_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fig = plt.figure()
ax = Axes3D(fig)
count = 0
x = []
y = []
z = []
for i in evaluate_acc_list:
    x.append(parameter_list[count % 10])
    y.append(parameter_list[math.floor(count / 10)])
    z.append(max(i))
    count += 1
# X, Y = np.meshgrid(np.array(x), np.array(y))
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax.scatter(x, y, z)
ax.set_zlabel('Z') 
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.savefig("./3D_evaluate_loss.jpg")
plt.show()


