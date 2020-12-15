import sys
import os
import time
import socket
import getopt
import openbabel

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
import Globals
import Convertor


# rdkit for chem-informatics
from rdkit import Chem
from rdkit.Chem import AllChem

def main(argv):
   Globals._init()

   # parameters to be used (IO later)
   QC_packages  =  ["G09"]
   Machines     =  ["ERA"]
   functionals  =  ["LC-BLYP"]
   bases        =  ["cc-pVTZ"]
   target_mols  =  ["./example/46507409.sdf"]
   ML_models    =  ["MPNN"]  # Maybe bug in MGCN

   Ncores       =  [1]

   usage_str='''example: python Fcst_kernel_A1.py -f|--func <functional> -b|--basis <basis> -i|--input <inputfile> -m|--model <model> -n|--ncores <ncores> -c|--cpu'''
   try:
      opts,args=getopt.getopt(argv[1:],
      "hcf:b:i:m:n:",
      ["help","cpu","func=","basis=","input=","model=","ncores="])
   except getopt.GetoptError:
      print(usage_str)
      sys.exit(2)


   for opt,arg in opts:
      if opt in ("-h","--help"):
         print(usage_str)
         sys.exit()
      elif opt in ("-c","--cpu"):
         print("force using cpu")
         Globals.set_value("cpu",True)
      elif opt in ("-f","--func"):
         functionals[0]=arg
      elif opt in ("-b","--basis"):
         bases[0]=arg
      elif opt in ("-i","--input"):
         #target_mols=arg.split(',')
         target_mols[0]=arg
      elif opt in ("-m","--model"):
         ML_models[0]=arg
      elif opt in ("-n","--ncores"):
         Ncores[0]=int(opt)


   # file format conversion
   informat=target_mols[0].split('.')[-1]
   if informat=="gjf":
      try:
         target_mols[0]=Convertor.GjfToSdf(target_mols[0])
      except Exception as e:
         print(e)
         return
   elif informat!='sdf':
      sdf_str_list=target_mols[0].split('.')
      sdf_str_list[-1]='sdf'
      sdf_str=".".join(sdf_str_list)
      obConversion = openbabel.OBConversion()
      obConversion.SetInAndOutFormats(informat, "sdf")
      inmol=openbabel.OBMol()
      try:
         obConversion.ReadFile(inmol, target_mols[0])
         obConversion.WriteFile(inmol,sdf_str)
         target_mols[0]=sdf_str
      except Exception:
         print("invalid input format of "+informat)
         return
      
      

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
                  ref_funct,ref_basis=DecideRefSpace.RefBasisfunct(basis,funct,mol)

                  print("  ===>   Target    Space : ", funct,"/",basis)
                  print("  ===>   Reference Space : ", ref_funct,"/",ref_basis)

                  # ref_chemspace = ref_funct + ref_basis 
                  ref_chemspace=ref_funct+"_"+ref_basis

                  # Predict basing on the ref_chemspace 
                  Ptime = PredictTime.Eval(mod,ref_chemspace,PWDmol,NAMmol,BAK,QC_packages[0],Machines[0]) 

                  # MWI correction for the predicted results
                  corr1 = PredictTime.MWIbasis(ref_chemspace,chemspace,PWDmol,NAMmol,PLYfile)
                  corr2 = PredictTime.MWIfunct(ref_chemspace,chemspace)

                  print("  ===>   The correction for funct/basis are ",corr2," and ",corr1," , respectively.")

                  Ptime=Ptime*corr1*corr2
                  print("  ===>   The predicted computational CPU time is ", Ptime)

            print("  ")
            print("  ")
   return
   
if __name__=="__main__":
   main(sys.argv)


