import sys
import os
import time
import socket

hostname = socket.gethostname()
PWD=os.getcwd()
SRC=PWD+"/src"
RAW=PWD+"/database/rawdata"
BAK=PWD+"/database/trained-models"

# add the runtime environments
print(SRC)
sys.path.append(SRC)

# originally import 
import torch
import Models
import Configs
import matplotlib.pyplot as plt
import numpy as np

ML_models   =  ["MPNN"]
TR_para     =  [25,250,50,0.01,1.0,2] # [NtrainSet,Nepoch,BatchSize,LRstep,TrainRatio,ValidInt]
TRM_dir      = PWD + "/database/training-models"
# SDFs and Crawled folder, related
sdfsH       = RAW + "/Arxiv1911.05569v1_sdfs_H"
setsDir     = RAW + "/G09data.01.updated"
# Functionals and basis sets, related
#functionals = ['B3LYP','bhandhlyp','BLYP','CAM-B3LYP','LC-BLYP','M06','M062x','PBE1PBE','wb97xd']
#bases       = ['6-31g','6-31gs','6-31pgs']
functionals = ['wb97xd']
bases       = ['6-31pgs']

# training/validing/testing sets
# ==> training sets, manual selections for adjusting the training models
suits_train = []
#suits_train.append("branch")
#suits_train.append("ring")
#suits_train.append("ring_sub")
#suits_train.append("alkane")
#suits_train.append("PE")
suits_train.append("Gaussian_inputs_training2")
suits_train.append("Gaussian_inputs_training")
suits_train.append("Gaussian_inputs_training3")
suits_train.append("Gaussian_inputs_training4")
suits_train.append("Gaussian_inputs_training5")

# ==> validing/testing sets, manual selections for adjusting 
suits_valid = []
suits_valid.append("Gaussian_inputs_validing")
suits_valid.append("Gaussian_inputs_validing2")
#suits_valid.append("Gaussian_inputs_testing")
#suits_valid.append("Gaussian_inputs_testing2")

# Training and validing/testing process

a = np.load('a.npy')
b = np.load('b.npy')
c = np.load('c.npy')
pic_dir = os.getcwd() + '/Result_b/mpnn_3'
if not os.path.exists(pic_dir):
    os.mkdir(pic_dir) 
pic_name = pic_dir + '/' + "wb97xd" + '.png'# + "_" + mol_size 
title = "MPNN_" + "wb97xd" #+ "_" + mol_size
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
plt.savefig(pic_name) 
plt.show()