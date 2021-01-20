import os

oldpar="SDF_INPUT"
newpar=""

folder="./SDF_LB"
for root,dirs,files in os.walk(folder):
   for file in files:
      print(file)
      newpar=file
      with open("Fcst_kernel_A1_SDF.py", 'w') as fw :
         with open("Fcst_kernel_A1_LB.py", 'r') as fr :
            lines=fr.readlines()
            for line in lines:
               newline=line.replace(oldpar,newpar)
               fw.write(newline) 
      os.system("python Fcst_kernel_A1_SDF.py")
exit(0)

