import sys
import os
import time
import re
#from numpy.random import randn

#nodes=input("how many nodes we use")
#print(nodes)

#prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631g/sdf_only_valid_atoms_working5/"
prefix="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631g/sdf_only_valid_atoms_working5_with_H/"
print("PREFIX : ", prefix)

#suits_file="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631g/data/valid_samples_20200602"
suits_file="/home/yingjin/Softwares/Fcst_sys/16-2-2_1to25_M062x-631gs/data/valid_samples_20200604"

with open("JMOL_script",'w') as jmol:
   jmol.close() 

with open(suits_file,'r') as suits:
   for line in suits:   
      tmpname=line.split()[4] 
      sdfname=prefix+tmpname+".sdf"
      print(sdfname)
      with open("JMOL_script",'a') as jmol:
         jmol.write("load FILES " + sdfname)
         jmol.write("\n")
         jmol.write("write IMAGE 200 200 PNG 2 " + tmpname + ".png")
         jmol.write("\n")

exit(0)

