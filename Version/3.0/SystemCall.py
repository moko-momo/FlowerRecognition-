import sys
import os

acc_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
outputPos = "./Output/"


for i in acc_list: 
    for j in acc_list:
        os.system("python KerasNetwork.py %f %f > %s" % (i, j, str(outputPos + str("out_%.1f_%.1f.log" % (i, j)))))