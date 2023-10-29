import sys
import os
import time
import socket

# originally import
import torch
import matplotlib.pyplot as plt
import numpy as np

data_path = os.getcwd() + '/eps/mpnn/size/improve1/'

a = np.load(data_path + 'MPNN_B3LYP_6-31pgs_small.npy')
b = np.load(data_path + 'MPNN_B3LYP_6-31pgs_middle.npy')
c = np.load(data_path + 'MPNN_B3LYP_6-31pgs_large.npy')
'''
a = np.load(data_path + 'MPNN_B3LYP_6-31g.npy')
b = np.load(data_path + 'MPNN_B3LYP_6-31gs.npy')
c = np.load(data_path + 'MPNN_B3LYP_6-31pgs.npy')
'''
pic_name = data_path + '/' + "B3LYP_mol_size" + '.eps'# + "_mol_size"
# title = "MPNN_" + "B3LYP" #+ "_" + mol_size
x = np.arange(0, 350)
# x1 = np.arange(0, 350)
# plt.title(title)
plt.xlabel("epoch",fontsize=15)
plt.ylabel("mre",fontsize=15)
plt.ylim((0, 1))
'''
plt.plot(x,a,label='6-31G')
plt.plot(x,b,label='6-31G*')
plt.plot(x,c,label='6-31+G*')
'''
plt.plot(x,a,label='small')
plt.plot(x,b,label='middle')
plt.plot(x,c,label='large')
plt.legend()
plt.grid()
plt.savefig(pic_name)
plt.show()