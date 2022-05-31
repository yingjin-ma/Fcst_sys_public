import sys
import os
import time
import socket

# originally import 
import torch
import matplotlib.pyplot as plt
import numpy as np


a = np.load('a.npy')
b = np.load('b.npy')
c = np.load('c.npy')
pic_dir = os.getcwd() + '/Result_b/mpnn_3'
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir) 
pic_name = pic_dir + '/' + "B3LYP" + '.png'# + "_" + mol_size 
title = "MPNN_" + "B3LYP" #+ "_" + mol_size
x = np.arange(0, 350)
x1 = np.arange(0, 350)
plt.title(title) 
plt.xlabel("epoch") 
plt.ylabel("mre")
plt.ylim((0, 1)) 
plt.plot(x,a,label='6-31g')
plt.plot(x,b,label='6-31gs')
plt.plot(x,c,label='6-31pgs')
plt.legend()
plt.grid()
#plt.savefig(pic_name) 
plt.show()