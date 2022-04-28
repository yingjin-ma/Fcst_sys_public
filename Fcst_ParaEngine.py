import sys
import os
import time
import socket
from os import listdir
from os.path import isfile, join

hostname = socket.gethostname()
PWD=os.getcwd()
SRC=PWD+"/src"
BAK=PWD+"/database/trained-models"
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
functionals  =  ["LC-BLYP"]
bases        =  ["cc-pVTZ"    ]
target_mols  =  ["./example/SDFs"]
ML_models    =  ["MPNN"]  # Maybe bug in MGCN

# rdkit treatment of input molecule

infiles = [f for f in listdir(target_mols[0]) if isfile(join(target_mols[0], f))]

mols   = []
NAMmol = []
for i in range(len(infiles)):
   tarfile = target_mols[0] + "/" + infiles[i]  
   mols.append(Chem.SDMolSupplier(tarfile))
   NAMmol.append(infiles[i])

#mol = mols[0]
#print(mols)

PWDmol = PWD + "/" + target_mols[0] 

print(" ")
print("PWDmol : ",PWDmol)
print("NAMmol : ",NAMmol)
print("BAKmod : ",BAK   )
print(" ")

#exit(0)

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
               ref_funct,ref_basis=DecideRefSpace.RefBasisfunct(basis,funct,mols[0][0])

               print("  ===>   Target    Space : ", funct,"/",basis)
               print("  ===>   Reference Space : ", ref_funct,"/",ref_basis)
               print(" ")

               # ref_chemspace = ref_funct + ref_basis 
               ref_chemspace=ref_funct+"_"+ref_basis

               for mol in mols:

                  # Predict basing on the ref_chemspace 
                  Ptime = PredictTime.EvalSuit(mod,ref_chemspace,PWDmol,NAMmol,BAK,QC_packages[0],Machines[0]) 
                  exit(0)

                  # MWI correction for the predicted results
                  corr1 = PredictTime.MWIbasis(ref_chemspace,chemspace,PWDmol,NAMmol,PLYfile)
                  corr2 = PredictTime.MWIfunct(ref_chemspace,chemspace)

                  print("  ===>   The correction for funct/basis are ",corr2," and ",corr1," , respectively.")
 
                  Ptime=Ptime*corr1*corr2
                  print("  ===>   The predicted computational CPU time is ", Ptime)

         print("  ")
         print("  ")

exit(0)


