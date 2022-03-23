import sys
import os
import time
import socket

hostname = socket.gethostname()
PWD=os.getcwd()
SRC=PWD+"/src"

is_local_model = input("Did you have trained locally? [y]/n:")
if is_local_model == "y":
   BAK=PWD+"/database/training-models"
else:
   BAK=PWD+"/database/training-models"
PLYfile=PWD+"/database/polyfitted/data.TOTpolyfitted.2"

# add the runtime environments
print(SRC)
sys.path.append(SRC)

# originally import 
import torch
import Models
import Configs
import PredictTime
import Magnification
import DecideRefSpace

# rdkit for chem-informatics
from rdkit import Chem
from rdkit.Chem import AllChem

# parameters to be used (IO later)
QC_packages  =  ["G09"]
Machines     =  ["ERA"]
functionals  =  ["B3LYP"]
bases        =  ["6-31g"]
target_mols  =  ["./example/404771460.sdf"]
ML_models    =  ["MPNN"]  # Maybe bug in MGCN

# rdkit treatment of input molecule
mols  =  [ mol for mol in Chem.SDMolSupplier(target_mols[0])]
mol   =  mols[0]

npath =  len(target_mols[0].split("/"))
if target_mols[0][0]=='/':
   PWDmol="/"
else: 
   PWDmol=""
for i in range(npath-1):
   PWDmol = PWD + "/" + target_mols[0].split("/")[i] 
NAMmol=  target_mols[0].split("/")[npath-1]

print("PWDmol : ",PWDmol)
print("NAMmol : ",NAMmol)
print("BAKmod : ",BAK   )

# chemical space and many-world interpretation for predicting

for qc in QC_packages:
   for imachine in Machines: 
      # QC_package@Machine  
      print("  ===>                                             " ) 
      print("  ===>   QC_package : ", qc, "     |     Machine : " , imachine) 
      print("  ===>                                             " ) 

      for mod in ML_models:    # models
         #Models.prepare(mod)

         print("  ")
         print("  =====================================================")
         print("  ===",mod,"===",mod,"===",mod,"===",mod,"===",mod,"===")
         print("  =====================================================")
         print("  ")

         for funct in functionals:   # functionals
            for basis in bases:      
               # ==   the target chemspace   == *  
               chemspace=funct+'_'+basis   

               # == decide the ref_chemspace == *
               # ref_funct & ref_basis
               ref_funct,ref_basis=DecideRefSpace.RefBasisfunct(basis,funct,mol,is_local_model)

               print("  ===>   Target    Space : ", funct,"/",basis)
               print("  ===>   Reference Space : ", ref_funct,"/",ref_basis)

               # ref_chemspace = ref_funct + ref_basis 
               ref_chemspace=ref_funct+"_"+ref_basis

               # Predict basing on the ref_chemspace 
               Ptime = PredictTime.Eval(mod,ref_chemspace,PWDmol,NAMmol,BAK,QC_packages[0],Machines[0]) 
               #print("  ===>   The predicted computational CPU time with no correction is ", Ptime)

               # MWI correction for the predicted results
               #corr1 = PredictTime.MWIbasis(ref_chemspace,chemspace,PWDmol,NAMmol,PLYfile)
               corr2 = PredictTime.MWIfunct(ref_chemspace,chemspace)

               #print("  ===>   The correction for funct/basis are ",corr2," and ",corr1," , respectively.")

               #Ptime=Ptime*corr1*corr2
               Ptime=Ptime*corr2
               print("  ===>   The predicted computational CPU time is ", Ptime)

         print("  ")
         print("  ")

exit(0)


