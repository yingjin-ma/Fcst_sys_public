import sys
import os
import time

os.system("find . -maxdepth 1 -type f -name 'runscrip*' > tmp")

submit_script="run_in_all"

with open("tmp", 'r+') as scripts:
   with open(submit_script,'w') as runscript:
      for runfile in scripts:
         runfile=runfile.replace('\n', '').replace('\r', '')
         print(runfile)
         runfile2=runfile+"-goon1"
         with open(runfile2,'w') as file2:
            file2.write("#!/bin/sh\n")
            file2.write("source ~/bash.g09\n")          
            with open(runfile,'r+') as file:
               with open("filetmp",'w') as ftmp:
                  for line in file:
                     if len(line.split()) > 1 :    
                        if line.split()[0] == "g09" :
                           filelog=line.split()[1].split("gjf")[0]+"log"
                           if os.path.isfile(filelog):
                              ftmp.seek(0,0)
                              ftmp.write(filelog + "               ")
                           else :
                              #print(filelog, " not exists")
                              file2.write("g09 "+filelog.split('log')[0]+"gjf\n")
               with open("filetmp",'r') as ftmp:
                  for line in ftmp:
                     file2.write("g09 "+line.split('log')[0]+"gjf\n")

#bsub -n 24 -q c_soft -o log.out -e log.err ./

         runscript.write("bsub -n 24 -q c_soft -o log.out -e log.err "+runfile2+"\n")  

os.system("chmod +x "+submit_script)

