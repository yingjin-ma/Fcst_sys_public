# In CLUSTER
import sys
import os
import time


oldpar="m062x/6-31g"
#newpar=[]

gau_dft  =["BLYP","LC-BLYP","B3LYP","CAM-B3LYP","M06","M062x","wb97xd","bhandhlyp","PBE1PBE"]
gau_basis=["6-31g","6-31g*","6-31+g*"]

gau_keywords=[]
for idft in gau_dft:
   for ibasis in gau_basis:
      gau_keywords.append(idft+"/"+ibasis)

gjf_folders=[]
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_training")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_training2")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_training3")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_training4")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_training5")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_validing")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_validing2")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_testing")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/Gaussian_inputs_testing2")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/PE")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/alkane")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/ring")
gjf_folders.append("/work1/scykd/mayj/Fcst_sys/16-2-2_1to25_M062x-631g/branch")


for newpar in gau_keywords:
       
   newpar_folder="Gaussian-"+newpar.replace("/","_").replace("*","s").replace("+","p")
   if not os.path.exists(newpar_folder):
      os.mkdir(newpar_folder)
   
   for folder in gjf_folders:
      nfolder=len(folder.split("/"))
      idir=folder.split("/")[nfolder-1]
#   print(folder) 
#   print(idir) 
#   exit(0)

      if not os.path.exists(idir):
         os.mkdir(idir)

      for root,dirs,files in os.walk(folder):
         for file in files:
            if len(file.split(".")) > 1 :
               if file.split(".")[1] == "gjf":
#               print(str(file))
                  filename=folder+"/"+file
                  filetmp =idir+"/"+file 
                  with open(filetmp , 'w') as fw :
                     with open(filename, 'r') as fr :
                        lines=fr.readlines()
#                        print(str(lines))
                        for line in lines:
                           newline=line.replace(oldpar,newpar)
                           fw.write(newline)

                  
      os.system("mv "+idir+" "+newpar_folder)
#               os.remove(filename)       
#               os.rename(filetmp,filename)       

      


