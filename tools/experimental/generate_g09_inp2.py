import sys
import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem

# check the cores, mem, scratch for Gaussian
gau_run  ="g09"
gau_cores="%nproc=24"
gau_mem  ="%mem=2000mw"
gau_title="scaling check"
# the calculation keywords
gau_keywords="#p m062x/6-31g pop=full nosym" # Series-1
#gau_keywords="#p m062x/6-31g* pop=full nosym" # Series-2
print (" ")
print ("Generated Gaussian inputs : ",gau_run,gau_cores,gau_mem,gau_keywords)
print (" ")

bypass=[]
#bypass.append("/home/yingjin/Softwares/Fcst_sys/SDFs_PE")

print("==============================================================================\n")
icount=0
oldinps=[]
print("The bypass Gaussian inputs folders :\n")
for ibypass in range(len(bypass)):
   print(bypass[ibypass])
   inp_list=os.listdir(bypass[ibypass]) 
   for inp in inp_list:
      if len(inp.split('.')) > 1 : 
         if inp.split('.')[1] == "gjf": 
            icount=icount+1 
            oldinps.append(inp.split('.')[0])

# Record all the generated inputs 
print("Number of previous Gaussian inputs : ",icount)
print("==============================================================================\n")

moldirs=[]
#moldirs.append("/home/yingjin/Softwares/Fcst_sys/SDFs_ring")
#moldirs.append("/home/yingjin/Softwares/Fcst_sys/SDFs_branch")
moldirs.append("/home/yingjin/Softwares/Fcst_sys/SDFs_ring_sub")

icount2=0
for imoldir in range(len(moldirs)):
   icount3=0
   moldir=moldirs[imoldir]
   mols=[]
   for root,dirs,files in os.walk(moldir):
      for f in files:
         mols.append(str(f))

   for m in mols:
      #if m.split(".")[1] == "sdf"
      if m.split(".")[1] == "sdf" or m.split(".")[1] == "mol":
         if m.split(".")[0] not in oldinps:
            icount3=icount3+1 
            output1=moldir+"/"+m
            print("output1 : ",output1)
            outputS=Chem.SDMolSupplier(output1)
            for mol in outputS:
#              print(mol.GetNumAtoms())
#              mol2=Chem.MolToSmiles(mol)
               m2 = Chem.AddHs(mol)
               AllChem.EmbedMolecule(m2)
               #outputT=moldir+"/"+"tmp.sdf"
               outputT=moldir+"/"+"tmp.sdf"
               print(Chem.MolToMolBlock(m2),file=open(outputT,'w+'))
#              print(m2.GetNumAtoms())
#              print(Chem.MolToMolBlock(m2))
#              exit()

               m1=m.split(".")[0]
               # convert mol file to the Gaussian input form
               output2=m1+'.gjf'
               os.system("obabel -i mol "+outputT+" -o gau -O "+output2)
               os.system("mv "+ outputT + " " + m1+'.sdf')

               # update the Gaussian input for calculating
               gau_chk="%chk="+m1+".chk"
               os.system("sed -i \'1 i "+gau_cores+"\' "+output2)
               os.system("sed -i \'2 i "+gau_mem  +"\' "+output2)
               os.system("sed -i \'3 i "+gau_chk  +"\' "+output2)
               os.system("sed -i \'4,5 d \' "+output2)
               os.system("sed -i \'4 i "+gau_keywords+"\' "+output2)
               os.system("sed -i \'6 i "+gau_title  +"\' "+output2)
               os.system("sed -i \'7d \' "+output2)

# generate the recording folder
#time_stamp_folder="result_"+time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
   stamp_folder =moldir + "/Gaussian_inputs-tmp"
   sdfs_folder  =moldir + "/SDFs_with_H"
   if not os.path.exists(stamp_folder):
      os.mkdir(stamp_folder)

   if not os.path.exists(sdfs_folder):
      os.mkdir(sdfs_folder)

   if icount3 >0 :
      os.system("mv *.gjf "+stamp_folder)
      os.system("mv *.sdf "+sdfs_folder)
   icount2=icount2+icount3

print("Totally ",icount2," non-redundant inputs are generated" )    

