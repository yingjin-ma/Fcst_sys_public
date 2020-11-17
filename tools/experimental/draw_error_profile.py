import sys
import os
import time
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#from numpy.random import randn

#nodes=input("how many nodes we use")
#print(nodes)

prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631g/"
#prefix="/home/yingjin/Documents/gau_prediction/figs2/"
#prefix="D:/FCST/fcst_sys/16-2-2_1to25_M062x-631g/"
print("PREFIX : ", prefix)

# Average CPU time
#result_RF_1  ="epoch1000-M600-LR0.0025-BS200"
result_RF_1="rf_ave_abcde_test.log"
result_LSTM_1="LSTM_test3"
#result_MPNN_1="MPNN_test2_E1000_LR0.001_BS25"
result_MPNN_1="mpnn_abcde_ave_test.log"
result_MGCN_1="MGCN_test1_E500_LR0.001_BS25"
# Total CPU time
#result_RF_2  ="epoch1000-M600-LR0.005-BS200"
result_RF_2="rf_tot_abcde_test.log"
result_LSTM_2="LSTM_tot_test2"
result_MPNN_2="mpnn_abcde_tot_test.log"
#result_MPNN_2="MPNN_tot_test1_E1000_LR0.001_BS25"
result_MGCN_2="MGCN_tot_test1_E500_LR0.001_BS25"

list_1=[]
list_1.append(result_RF_1)
list_1.append(result_LSTM_1)
list_1.append(result_MPNN_1)
list_1.append(result_MGCN_1)
list_2=[]
list_2.append(result_RF_2)
list_2.append(result_LSTM_2)
list_2.append(result_MPNN_2)
list_2.append(result_MGCN_2)

# extract the results
time_1T=[]
time_1P=[]
error_1=[]
dv1=0.0
for result in list_1:
   if result =="" : 
      print("No result and use 0.0 instead") 
   else:
      filename=prefix+result 
      print(filename) 
      with open(filename,'r',encoding="utf-8") as f:
         lines=f.readlines() 
         for line in lines:
            imatch_list=re.search("i:",line)
            if imatch_list !=None:  
               time_1T.append(float(line.split()[10]))
               time_1P.append(float(line.split()[13]))
               dtmp = float(line.split()[13])-float(line.split()[10])
               dtmp = dtmp/float(line.split()[10])
               error_1.append(dtmp)

time_1T=[]
time_1P=[]
error_2=[]
for result in list_2:
   if result =="" : 
      print("No result and use 0.0 instead") 
   else:
      filename=prefix+result 
      print(filename) 
      with open(filename,'r',encoding="utf-8") as f:
         lines=f.readlines() 
         for line in lines:
            imatch_list=re.search("i:",line)
            if imatch_list !=None:  
               time_1T.append(float(line.split()[10]))
               time_1P.append(float(line.split()[13]))
               dtmp = float(line.split()[13])-float(line.split()[10])
               dtmp = dtmp/float(line.split()[10])
               error_2.append(dtmp)

#print(time_1T)
#print(time_1P)
print(len(error_1))
#print("part",(error_1[0:2]))
print(len(error_2))

errpro1 = [[] for i in range(4)] 
errpro2 = [[] for i in range(4)] 

for imod in range(4):
   for i in range(5):
      for j in range(5):
         ij=j*5+i+imod*25 
         errpro1[imod].append(error_1[ij])
         errpro2[imod].append(error_2[ij])

print(errpro1[0])
print(errpro1[1])
print(errpro1[2])
print(errpro1[3])

for imod in range(len(errpro1)):
   dv=0
   dz=0
   for i in range(len(errpro1[imod])):
      dv =dv  + abs(errpro1[imod][i])
      dz =dz  + abs(errpro2[imod][i])
   dv=dv/len(errpro1[imod])
   dz=dz/len(errpro1[imod])
   print("TOT ",dv," AVE ",dz)        

for imod in range(len(errpro1)):
   dv=0
   dz=0
   print("imod same basis num ",imod)
   for i in range(len(errpro1[imod])):
      dv =dv  + abs(errpro1[imod][i])
      dz =dz  + abs(errpro2[imod][i])
      if (i+1)%5 == 0 :
        dv=dv/5
        dz=dz/5
        print("TOT ",dv," AVE ",dz)        
        dv=0.0
        dz=0.0

for imod in range(len(errpro1)):
   dv=0
   dz=0
   print("imod same type ",imod)
   for i in range(5):
      dv =dv  + abs(errpro1[imod][i]) + abs(errpro1[imod][i+5]) + abs(errpro1[imod][i+10]) + abs(errpro1[imod][i+15]) + abs(errpro1[imod][i+20])
      dz =dz  + abs(errpro2[imod][i]) + abs(errpro2[imod][i+5]) + abs(errpro2[imod][i+10]) + abs(errpro2[imod][i+15]) + abs(errpro2[imod][i+20])
      dv=dv/5
      dz=dz/5
      print("TOT ",dv," AVE ",dz)        
      dv=0.0
      dz=0.0


labels=['1A','1B','1C','1D','1E','2A','2B','2C','2D','2E','3A','3B','3C','3D','3E','4A','4B','4C','4D','4E','5A','5B','5C','5D','5E']
labels1=[-1.0,-0.75,-0.50,-0.25,0.0,0.25,0.50,0.75,1.0]
labels2=['-1.00','-0.75','-0.50','-0.25','0.00','0.25','0.50','0.75','1.00']

fig=plt.figure()

ax1=fig.add_subplot(4,1,1)
ax1.bar(np.arange(len(errpro1[0]))+0,errpro1[0], alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='Tot. CPU time', lw=1)
ax1.bar(np.arange(len(errpro1[0]))+0.35, errpro2[0], alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='Ave. CPU time', lw=1)
plt.legend(loc="upper right")
plt.ylim(ymin=-1,ymax=1)
plt.xticks(np.arange(len(errpro1[0])),labels)
plt.yticks(labels1,labels2)

ax2=fig.add_subplot(4,1,2)
ax2.bar(np.arange(len(errpro1[1]))+0,errpro1[1], alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='Tot. CPU time', lw=1)
ax2.bar(np.arange(len(errpro1[1]))+0.35, errpro2[1], alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='Ave. CPU time', lw=1)
plt.legend(loc="upper right")
plt.ylim(ymin=-1,ymax=1)
plt.xticks(np.arange(len(errpro1[0])),labels)
plt.yticks(labels1,labels2)

ax3=fig.add_subplot(4,1,3)
ax3.bar(np.arange(len(errpro1[2]))+0,errpro1[2], alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='Tot. CPU time', lw=1)
ax3.bar(np.arange(len(errpro1[2]))+0.35, errpro2[2], alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='Ave. CPU time', lw=1)
plt.legend(loc="upper right")
plt.ylim(ymin=-1,ymax=1)
plt.xticks(np.arange(len(errpro1[0])),labels)
plt.yticks(labels1,labels2)

ax4=fig.add_subplot(4,1,4)
ax4.bar(np.arange(len(errpro1[3]))+0,errpro1[3], alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='Tot. CPU time', lw=1)
ax4.bar(np.arange(len(errpro1[3]))+0.35, errpro2[3], alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='Ave. CPU time', lw=1)
plt.legend(loc="upper right")
plt.ylim(ymin=-1,ymax=1)
plt.xticks(np.arange(len(errpro1[0])),labels)
plt.yticks(labels1,labels2)

plt.show()

exit(0)


