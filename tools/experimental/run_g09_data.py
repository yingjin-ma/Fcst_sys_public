import sys
import os
import time
import re

#nodes=input("how many nodes we use")
#print(nodes)

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

for mdir in moldir:

   mols=[]
   for root,dirs,files in os.walk(mdir):
      for f in files:
         mols.append(str(f))

   datafile="data/"+str(mdir)

#   print("data")
   if not os.path.exists("data"):
      os.mkdir("data")
   print(datafile)
#   if not os.path.exists(datafile):
#      os.mkdir(datafile)

   ii=0
   with open(datafile,'w') as f:
      f.write("")

   for m in mols:
      print(m)
      if len(m.split(".")) > 1 :
         if m.split(".")[1] == "gjf":
            ii=ii+1
            glog=mdir+"/"+m.split(".")[0]+'.log' 
            name=m.split('.')[0]
            cpu_time=[] # contains 4 elements:days,hours,minutes,seconds
            elp_time=[] # same as cpu_time
            delta_E =[] # same as cpu_time
            basis   =[] # number of basis functions
            electron=[] # number of electrons
            function="" # memory amount
            bas_name="" # memory amount
            maxmem  = 0  # memory amount
            ncycle  = 0 
            itermination = -1
            with open(glog,'r') as log:
               for line in log:
                  imatch_termn = line.find("Normal termination")
                  imatch_time1 = re.match(" Job cpu time",line)
                  imatch_time2 = re.match(" Elapsed time",line)
                  imatch_cycle = re.match(" Cycle",line)
                  imatch_delta = line.find("Delta-E=")
                  imatch_elect = line.find("alpha electrons")
                  imatch_mem   = line.find("MaxMem=")
                  imatch_fun   = line.find("SCF Done:")
                  imatch_basis = line.find("Standard basis:")
                  if imatch_termn != -1 :
                     itermination = 1
                  if imatch_basis != -1 :
                     bas_name=line.split()[2]
                  if imatch_fun != -1 :
                     function=line.split()[2]
                  if imatch_mem != -1 :
                     for i in range(len(line.split())):
                        if line.split()[i] == "MaxMem=":
                           if maxmem < (line.split()[i+1]):
                              maxmem = long(line.split()[i+1])
                              #print(maxmem)
                  if imatch_delta != -1 :
                     delta_E.append(line.split()[3])
                  if imatch_cycle != None :
                     ncycle=ncycle+1 
                  if imatch_time1 != None :
                     for i in range(3,10,2):
                        cpu_time.append(float(line.split()[i]))
                  if imatch_time2 != None :
                     for i in range(2,9,2):
                        elp_time.append(float(line.split()[i]))
                  #print(imatch_cycle)
                  if imatch_elect != -1 :
                     electron.append(int(line.split()[0]))
                     electron.append(int(line.split()[3]))
                  imatch_basis=re.search(" basis functions",line)
                  if imatch_basis !=None:
#                  print(line)
                     try:
                        basis.append(int(line.split()[0])) #
                        basis.append(int(line.split()[0])) #
                     except Exception:
                        print("no result -- Exception")                      

            print("itermination : ",itermination)
 
            if len(cpu_time)<4 or len(elp_time)<4:
               print("no result ") 

#           time_elp=24*3600*elp_time[0]+3600*elp_time[1]+60*elp_time[2]+elp_time[3]  

#            for i in length(delta_E):
#               print(delta_E[i]) 
#            print(delta_E) 

            if itermination ==1 :
               if basis[0] > 0 and itermination ==1 :
                  time_cpu=24*3600*cpu_time[0]+3600*cpu_time[1]+60*cpu_time[2]+cpu_time[3]
                  with open(datafile,'a+') as f:
                     line = str(basis[0])+"  0  " + str(time_cpu) + "  0  " + str(name) + "  " + str(ncycle) + "  " + str(time_cpu/float(ncycle))  + "  " +  str(maxmem) + "  " + bas_name + "  " + str(basis) + "  " + function  + "  "  + str(electron) + "  " + str(delta_E) + "\n"
                     f.write(line) 
            else :
               print("Error terminationed, bypass ")



