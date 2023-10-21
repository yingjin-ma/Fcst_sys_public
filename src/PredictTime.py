import sys
import os
import time
import socket

import Configs
import torch
import MpnnToolEVAL
import Models
import JacobLadder

from Magnification import getNbasis
from Magnification import fitted_magns

def Eval(model,ref_chemspace,PWDmol,NAMmol,BAK,QC_packages,Machines):

   model=model.upper()    
   #print("PWDmol : ",PWDmol)
   
   if model=="MPNN":
      import MpnnToolEVAL  

      # Parameters
      torch.manual_seed(2) 
      training_size = 2000
      aimming       = 2
      modelName     = Models.ModelLoad(model,BAK,ref_chemspace,QC_packages,Machines)
      #modelName = os.getcwd() + "/database/training-modelsmpnn_B3LYP_6-31g_tot.pkl"
      # Step-1 : establish the configuration
      config   = Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,batch_size=200,tra_set_ratio=1,valid_interval=2)
      # step-2 : initialize the MPNN models 
      MPNNtool = MpnnToolEVAL.MpnnTool(chemspace=ref_chemspace,config=config,suits1=NAMmol,sdf_dir=PWDmol,target=aimming)
      # step-3 : get the predicted results
      RES_tot  = MPNNtool.eval(modelname=modelName,path=PWDmol,mol=NAMmol,chemspace=ref_chemspace)   

      Ptime=RES_tot[0]

   elif model=="MGCN": 
      import MgcnToolEVAL 

      # Parameters
      torch.manual_seed(2) 
      training_size = 2000
      aimming       = 2
      modelName     = Models.ModelLoad(model,BAK,ref_chemspace,QC_packages,Machines)

      # Step-1 : establish the configuration
      config   = Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,batch_size=200,tra_set_ratio=1,valid_interval=2)
      # step-2 : initialize the MGCN models 
      MGCNtool = MgcnToolEVAL.MgcnTool(chemspace=ref_chemspace,config=config,suits1=NAMmol,sdf_dir=PWDmol,target=aimming)
      # step-3 : get the predicted results
      RES_tot  = MGCNtool.eval(modelname=modelName,path=PWDmol,mol=NAMmol,chemspace=ref_chemspace)   

      Ptime=RES_tot[0]

   elif model=="LSTM": 
      import LstmToolEVAL 

      # Parameters
      torch.manual_seed(2) 
      training_size = 2000
      aimming       = 2
      modelName     = Models.ModelLoad(model,BAK,ref_chemspace,QC_packages,Machines)

      # Step-1 : establish the configuration
      config   = Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,batch_size=200,tra_set_ratio=1,valid_interval=2)
      # step-2 : initialize the LSTM models 
      LSTMtool = LstmToolEVAL.LstmTool(chemspace=ref_chemspace,config=config,suits1=NAMmol,sdf_dir=PWDmol,target=aimming)
      # step-3 : get the predicted results
      RES_tot  = LSTMtool.eval(modelname=modelName,path=PWDmol,mol=NAMmol,chemspace=ref_chemspace,BAK=BAK)   

      Ptime=RES_tot[0]

   elif model=="RF": 
      import RfToolEVAL 

      # Parameters
      torch.manual_seed(2) 
      training_size = 2000
      aimming       = 2
      modelName     = Models.ModelLoad(model,BAK,ref_chemspace,QC_packages,Machines)

      # Step-1 : establish the configuration
      config   = Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,batch_size=200,tra_set_ratio=1,valid_interval=2)
      # step-2 : initialize the   RF models 
      RFtool = RfToolEVAL.RfTool(chemspace=ref_chemspace,config=config,suits1=NAMmol,sdf_dir=PWDmol,target=aimming,BAK=BAK)
      # step-3 : get the predicted results
      RES_tot  = RFtool.eval(modelname=modelName,path=PWDmol,mol=NAMmol,chemspace=ref_chemspace,BAK=BAK)   

      Ptime=RES_tot[0]


   return Ptime

def MWIbasis(ref_chemspace,chemspace,PWDmol,NAMmol,PLYfile):

   molecule=PWDmol+"/"+NAMmol
   print(molecule)

   refbasis = ref_chemspace.split("_")[1]
   reffunct = ref_chemspace.split("_")[0]
   tarbasis =     chemspace.split("_")[1]
   tarfunct =     chemspace.split("_")[0]

   nbasis_ref = getNbasis(bas=refbasis,sdf=molecule) 
   nbasis_tar = getNbasis(bas=tarbasis,sdf=molecule) 

   dv_magn    = fitted_magns([nbasis_ref],[nbasis_tar],ref_chemspace,ployfitted=PLYfile) 

   print("nbasis_ref",nbasis_ref, "nbasis_tar",nbasis_tar,"dv_magns",dv_magn)

   return dv_magn[0]


def MWIfunct(ref_chemspace,chemspace):

   dv_magn=1.0

   refbasis = ref_chemspace.split("_")[1]
   reffunct = ref_chemspace.split("_")[0]
   tarbasis =     chemspace.split("_")[1]
   tarfunct =     chemspace.split("_")[0]

   iladder1=JacobLadder.position(reffunct)
   iladder2=JacobLadder.position(tarfunct)
  
   #print("iladder1 : ",iladder1,"iladder2 : ",iladder2) 
   if iladder1 == iladder2:
      dv_magn=1.0
   elif iladder1 > iladder2: 
      if iladder1-iladder2 >= 2:
         dv_magn=0.90
      else:
         dv_magn=0.95 
   else:
      if iladder2-iladder1 >= 2: 
         dv_magn=1.2 
      else:
         dv_magn=1.1 
           

   return dv_magn

