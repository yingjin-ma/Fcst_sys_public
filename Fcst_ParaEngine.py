import sys
import os
import time
import socket
from os import listdir
from os.path import isfile, join

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
sys.path.append(PWD+"/tools")

# originally import 
import torch
import Models
import Configs
import PredictTime
import Magnification
import DecideRefSpace
import SplitPDB

# rdkit for chem-informatics
from rdkit import Chem
from rdkit.Chem import AllChem


def add_nitrogen_charges(smiles):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m


# parameters to be used (IO later)
QC_packages  =  ["G09"]
Machines     =  ["ERA"]
functionals  =  ["M062x"]
bases        =  ["6-31pgs"]
#target_mols  =  ["./example/Arxiv1911.05569v1_sdfs_H_Part2"]
target_PDB   =  ["./example/LBtest29-6-80A_para.pdb"]
#target_mols  =  ["./updatedSDFs"]
#target_smil  =  ["./TestSMI2"]
ML_models    =  ["MPNN"]  # Maybe bug in MGCN

ifmols = False
ifsmil = False

if "target_PDB" in dir():
    SplitPDB.split(target_PDB[0],"./tmpPDB")
    print("target_PDB  is defined")
    target_mols  = ["./tmpPDB"]
    infiles = [f for f in listdir(target_mols[0]) if isfile(join(target_mols[0], f))]
    ifmols = True

if "target_smil" in dir():
    print("target_smil  is defined")
    infiles = [f for f in listdir(target_smil[0]) if isfile(join(target_smil[0], f))]
    ifsmil = True

if "target_mols" in dir():
    print("target_mols  is defined")
    infiles = [f for f in listdir(target_mols[0]) if isfile(join(target_mols[0], f))]
    ifmols = True

# rdkit treatment of input molecule

mols   = []
NAMmol = []
for i in range(len(infiles)):

   if ifmols:
       tarfile = target_mols[0] + "/" + infiles[i]  
       print(i, " tarfile : ", tarfile)
       mols.append(Chem.SDMolSupplier(tarfile))
       PWDmol = PWD + "/" + target_mols[0] 
       NAMmol.append(infiles[i])

   if ifsmil:
       tarfile = target_smil[0] + "/" + infiles[i]  
       print(i, " tarfile : ", tarfile)

       with open(tarfile, 'r') as fsmiles:
           lines = fsmiles.readlines()

       #print("lines : ", lines)    
       for line in lines:
           ismi  = line.split()[0]
           iname = line.split()[1]
           print("ismi : ", ismi )
           msmi = add_nitrogen_charges(ismi)
           ismi2= Chem.MolToSmiles(msmi) 
           qmol  = Chem.MolFromSmiles(ismi2)
           print("qmol : ", qmol) 
           #qmol.UpdatePropertyCache(strict=False)
           mols.append(qmol)
           NAMmol.append(iname)

       PWDmol = PWD + "/" + target_smil[0] 


#mol = mols[0]
#print(mols)


print(" ")
#print("PWDmol : ",PWDmol)
#print("NAMmol : ",NAMmol)
#print("BAKmod : ",BAK   )
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

               #print("mols : ",mols[0])

               # == decide the ref_chemspace == *
               # ref_funct & ref_basis
               ref_funct,ref_basis=DecideRefSpace.RefBasisfunct(basis,funct,mols[0][0],is_local_model)
               #ref_funct,ref_basis=DecideRefSpace.RefBasisfunct(basis,funct,mols[0])

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


