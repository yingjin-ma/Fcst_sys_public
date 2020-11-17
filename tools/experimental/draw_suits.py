import sys
import os
import time
import re
import matplotlib.pyplot as plt
#from numpy.random import randn

#nodes=input("how many nodes we use")
#print(nodes)

prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631gs/data/"
print("PREFIX : ", prefix)

moldir=[]
moldir.append("PE")
moldir.append("alkane")
moldir.append("branch")
moldir.append("ring")
moldir.append("ring_sub")
moldir.append("Gaussian_inputs_training")
moldir.append("Gaussian_inputs_training2")
moldir.append("Gaussian_inputs_training3")
moldir.append("Gaussian_inputs_training4")
moldir.append("Gaussian_inputs_training5")
moldir.append("Gaussian_inputs_testing")
moldir.append("Gaussian_inputs_testing2")
moldir.append("Gaussian_inputs_validing")
moldir.append("Gaussian_inputs_validing2")

metadata=[]
for mdir in moldir:

   mols=[]
   for root,dirs,files in os.walk(mdir):
      for f in files:
         mols.append(str(f))

   datafile=prefix+str(mdir)

   if not os.path.exists("data"):
      os.mkdir("data")

   ii=0
   with open(datafile,'r') as f:
      metadata.append(f.readlines())

#print(metadata[0][0].split())

with open("testing_suits",'w') as suit:
   suit.close() 

basis=[]
cpu_ave=[]
cpu_tot=[]
for i in range(len(metadata)):
   for j in range(len(metadata[i])):
      if int(metadata[i][j].split()[0]) <= 900: 
         basis.append(int(metadata[i][j].split()[0]))  
         cpu_tot.append(float(metadata[i][j].split()[2])) 
         cpu_ave.append(float(metadata[i][j].split()[6])) 
         ibasis=int(metadata[i][j].split()[0])
#      for k in range(len(aiming_basis)):
#         if abs(ibasis - aiming_basis[k]) < nfluct:
##            print(i,j,ibasis,metadata[i][j].split()[4])    
#            recording[k]=recording[k]+1
#            if recording[k] < 4: #
#               print(metadata[i][j])
#               with open("testing_suits",'a') as suit:
#                  line=metadata[i][j] 
#                  suit.write(line) 

#print(basis)
fig=plt.figure()

ax1=fig.add_subplot(3,1,1)
ax1=ax1.hist(basis,bins=1500)
plt.xlabel('N basis')
plt.ylabel('Count')
ax2=fig.add_subplot(3,1,2)
ax2=ax2.hist(cpu_ave,bins=1500,color="yellowgreen")
plt.xlabel('Ave. CPU Time (s)')
plt.ylabel('Count')
ax2=fig.add_subplot(3,1,3)
ax2=ax2.hist(cpu_tot,bins=1500,color="lightskyblue")
plt.xlabel('Tot. CPU Time (s)')
plt.ylabel('Count')
#plt.xlim(xmin=0, xmax=5000)

plt.show()
 
#print(metadata)
#print(metadata[0][0])  

