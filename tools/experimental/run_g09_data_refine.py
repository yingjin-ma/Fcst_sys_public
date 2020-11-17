import sys
import os
import time
import re

#prefix=input("Where are the data?")
prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631gs/data/"
#prefix="/home/yingjin/Softwares/Fcst_sys/Spaces/Gaussian-B3LYP_6-31g/data/"
print(prefix)

#nbasis_limit=[0,600]
nbasis_limit=[0,900]

moldir=[]
moldir.append("Gaussian_inputs_training")
moldir.append("Gaussian_inputs_training2")
moldir.append("Gaussian_inputs_training3")
moldir.append("Gaussian_inputs_training4")
moldir.append("Gaussian_inputs_training5")
moldir.append("Gaussian_inputs_validing")
moldir.append("Gaussian_inputs_validing2")
moldir.append("Gaussian_inputs_testing")
moldir.append("Gaussian_inputs_testing2")
moldir.append("PE")
moldir.append("alkane")
moldir.append("ring")
moldir.append("branch")
moldir.append("ring_sub")

for mdir in moldir:

   mols=[]
   for root,dirs,files in os.walk(mdir):
      for f in files:
         mols.append(str(f))

   datafile =prefix+str(mdir)
   datafile2=prefix+str(mdir)+".re"

   ii=-1
   print(datafile)

   if(not os.path.exists(datafile)):
     print(" ========> Can not find the  ",datafile, " <========= " )
   else :    
     print("Refined as ",datafile2)  
     ii=0

   if ii == 0:
      with open(datafile2,'w') as redata:
         with open(datafile,'r') as rawdata:
            for line in rawdata:
               nbasis=int(line.split()[0]) 
               if nbasis > nbasis_limit[0] and nbasis < nbasis_limit[1]:
                  redata.write(line)  

