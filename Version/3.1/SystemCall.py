import sys
import os


acc_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
outputPos = "./Output/"
conv_num = 128
dense_num = 512

for i in acc_list: 
    for j in acc_list:
        os.system("python KerasNetwork.py %d %f %f %d > %s" % (conv_num, i, j, dense_num, str(outputPos + str("out_%.1f_%.1f.log" % (i, j)))))