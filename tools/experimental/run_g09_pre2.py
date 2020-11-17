# In CLUSTER
import sys
import os
import time

nodes=input("how many nodes we use")
print(nodes)
os.system("rm runscript-*")

moldir="./"

mols=[]
for roots,dirs,files in os.walk(moldir):
   for f in files:
      mols.append(str(roots)+"/"+str(f))

ii=0
for m in mols:
    #print(m)
    if len(m.split(".")) > 2 :
       if m.split(".")[2] == "gjf":
          ii=ii+1
          imod=ii%int(nodes)
          runfile="runscript-"+str(imod)
          with open(runfile, 'a+') as f:
             runline="g09 "+ m  
             print(runline)
             f.write(runline) 
             f.write("\n") 

for i in range(int(nodes)):
   runfile="runscript-"+str(i) 
   with open(runfile, 'r+') as f:
      content = f.read()        
      f.seek(0, 0)
      f.write("#!/bin/sh\n")
      f.write("source ~/bash.g09\n")
      f.write(content)
   os.system("chmod +x "+runfile)
