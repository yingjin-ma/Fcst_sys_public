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

#prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631g/"
# prefix="D:/Fcst/fcst_sys/16-2-2_1to25_M062x-631g/"

prefix="C:/codes/Fcst_sys/16-2-2_1to25_M062x-631g/"
#prefix="/home/yingjin/Documents/gau_prediction/figs2/"
print("PREFIX : ", prefix)

# Average CPU time
result_LSTM_ave=["LSTM_ave_onlyRFmoldes_BS100_test1_valid-5row","Lstm_ave_RFmodels+DB_BS100_M300_test1_0602","Lstm_ave_RFmodels+DB_BS100_M500_test1_0602","LSTM_RFmodels+DB_BS100_M700_test1_0602","LSTM_ave_RFmoldes+DB_BS100_M900_test2_valid-5row","LSTM_ave_RFmoldes+DB_BS100_M1200_test2_valid-5row","LSTM_ave_RFmoldes+DB_BS100_M1500_test2_valid-5row"]
result_MPNN_ave=["MPNN_ave_onlyRFmoldes_BS100_test1_valid-5row","MPNN_ave_RFmodels+DB_BS100_M300_test1_0602","MPNN_ave_RFmodels+DB_BS100_M500_test1_0602","MPNN_ave_RFmoldes+DB_BS100_M700_test1_valid-5row","MPNN_ave_RFmoldes+DB_BS100_M900_test1_valid-5row","MPNN_ave_RFmoldes+DB_BS100_M1200_test2_valid-5row","MPNN_ave_RFmoldes+DB_BS100_M1500_test1_valid-5row"]
result_MGCN_ave=["MGCN_ave_onlyRFmoldes_BS100_test1_valid-5row","MGCN_ave_RFmoldes+DB_BS100_M300_test1_valid-5row","MGCN_test1_E500_LR0.001_BS25","MGCN_ave_RFmoldes+DB_BS100_M700_test1_valid-5row","MGCN_ave_RFmoldes+DB_BS100_M900_test1_valid-5row","MGCN_ave_RFmoldes+DB_BS100_M1200_test1_valid-5row","MGCN_ave_RFmodels+DB_BS100_M1500_test1_0602"]
# Total CPU time
result_LSTM_tot=["LSTM_onlyRFmoldes_test1_valid-5row","LSTM_RFmoldes+DB_BS100_M300_test2_valid-5row","LSTM_tot_test2","LSTM_RFmodels+DB_BS100_M700_test1_0602","LSTM_RFmoldes+DB_BS100_M900_test3_valid-5row","LSTM_RFmoldes+DB_BS100_M1200_test2_valid-5row","LSTM_RFmoldes+DB_BS100_M1500_test2_valid-5row"]
result_MPNN_tot=["MPNN_onlyRFmoldes_BS100_test1_valid-5row","MPNN_RFmoldes+DB_BS100_M300_test4_valid-5row","MPNN_test2_E1000_LR0.001_BS25","MPNN_RFmoldes+DB_BS100_M700_test1_valid-5row","MPNN_RFmoldes+DB_BS100_M900_test1_valid-5row","MPNN_RFmodels+DB_BS100_M1200_test1_0602","MPNN_RFmoldes+DB_BS100_M1500_test1_valid-5row"]
result_MGCN_tot=["MGCN_onlyRFmoldes_BS100_test1_valid-5row","MGCN_RFmoldes+DB_BS100_M300_test1_valid-5row","MGCN_tot_test1_E500_LR0.001_BS25","MGCN_RFmoldes+DB_BS100_M700_test1_valid-5row","MGCN_RFmoldes+DB_BS100_M900_test1_valid-5row","MGCN_RFmodels+DB_BS100_M1200_test3_0602","MGCN_RFmoldes+DB_BS100_M1500_test1_valid-5row"]

list_ave=[result_LSTM_ave,result_MPNN_ave,result_MGCN_ave]
list_tot=[result_LSTM_tot,result_MPNN_tot,result_MGCN_tot]

#print(len(list_ave))
#print(list_ave[0][0])
#exit(0)

RE_tot = [[] for i in range(len(list_tot))]
# extract the results
for i in range(len(list_tot)):
   for result in list_tot[i]:
      time_1T=[]
      time_1P=[]
      error_1=[]
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
      RE_tot[i].append(error_1)               


RE_ave = [[] for i in range(len(list_ave))]
for i in range(len(list_ave)):
   for result in list_ave[i]:
      time_1T=[]
      time_1P=[]
      error_2=[]
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
      RE_ave[i].append(error_2)               
                  
#print(RE_tot[0][0][0])
#print(RE_ave[0])
#print(RE_ave[0][0])
#print(RE_ave[0][0][0])

errpro1 = [[] for i in range(len(list_tot))] 

for imod in range(len(list_tot)):
   for i in range(len(list_tot[imod])):
      dv=0.0 
      for j in range(len(RE_tot[imod][i])):
         dv=dv+abs(RE_tot[imod][i][j])
      dv=dv/len(RE_tot[imod][i])   
      errpro1[imod].append(dv)
   print(errpro1[imod])   
     
#print(list_ave)
print(list_ave[0][0])
errpro2 = [[] for i in range(len(list_ave))] 

print(imod,i,len(RE_ave[0][0]))
for imod in range(len(list_ave)):
   for i in range(len(list_ave[imod])):
#      print(imod,i,len(RE_ave[imod][i])) 
      dv=0.0
      for j in range(len(RE_ave[imod][i])):
         dv=dv+abs(RE_ave[imod][i][j])
      dv=dv/len(RE_ave[imod][i])
      errpro2[imod].append(dv)
   print(errpro2[imod])

#         print(imod,i,len(RE_tot[imod][i])) 
#         print(imod,i,len(RE_tot[imod][i])) 

#      for j in range(len(list_tot[imod][i])):
#         print("imodel : ",imod,"; i-th train : ",i,"; j-th sample : ",j) 

xdim=[92,300,500,700,900,1200,1500]


fontsize=14
fontdict={'size':fontsize}

fig=plt.figure()

ax1=fig.add_subplot(1,2,1)
ax1.plot(xdim,errpro1[0],color='red',linewidth=3.0,linestyle='-',label="LSTM")
ax1.plot(xdim,errpro1[1],color='blue',linewidth=3.0,linestyle='-.',label="MPNN")
ax1.plot(xdim,errpro1[2],color='green',linewidth=3.0,linestyle='--',label="MGCN")
plt.legend(loc="upper right",prop=fontdict)
plt.xlabel('Number of molecules in training suit',fontdict=fontdict)
plt.ylabel('MRE of total CPU times',fontdict=fontdict)
#plt.title('Tot. CPU times')
plt.ylim(ymin=0,ymax=1.0)

# 设置刻度字体大小
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)


ax2=fig.add_subplot(1,2,2)
ax2.plot(xdim,errpro2[0],color='red',linewidth=3.0,linestyle='-',label="LSTM")
ax2.plot(xdim,errpro2[1],color='blue',linewidth=3.0,linestyle='-.',label="MPNN")
ax2.plot(xdim,errpro2[2],color='green',linewidth=3.0,linestyle='--',label="MGCN")
plt.legend(loc="upper right",prop=fontdict)
plt.xlabel('Number of molecules in training suit',fontdict=fontdict)
plt.ylabel('MRE of average CPU times',fontdict=fontdict)
#plt.title('Ave. CPU times')
plt.ylim(ymin=0,ymax=1.0)

# 设置刻度字体大小
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.show()

exit(0)

