import os
import sys
import time
import socket

import Similarity
import JacobLadder

hostname = socket.gethostname()
PWD=os.getcwd()
BASE=PWD+"/database"


def getknownbasisfunct(is_local_model):

    TRmodels=[]
    if is_local_model == "y":
      TRAINED=BASE+"/training-models"
    else:
      TRAINED=BASE+"/trained-models"
    
    print(TRAINED)
    for roots,dirs,files in os.walk(TRAINED):
       #print(roots," ++ ",dirs) 
       TRmodels=dirs
       break
#      print(TRmodels)
    
    TRbasis0=[]
    TRfunct0=[]
    for imod in TRmodels:
       TRbasis0.append(imod.split("_")[3])
       TRfunct0.append(imod.split("_")[2])

# remove the redundants
    TRbasis=list(set(TRbasis0)) 
    #{}.fromkeys(TRbasis0).keys() ## 'dict_keys' object 
    TRfunct=list(set(TRfunct0)) 
    #{}.fromkeys(TRfunct0).keys() ## 'dict_keys' object

#    print(TRbasis)        
#    print(TRfunct)    
    
    return TRbasis,TRfunct

def RefBasisfunct(basis,funct,mol,is_local_model):

    # obtain the trained basis & functional 
    TRbasis,TRfunct=getknownbasisfunct(is_local_model)


    # || BASIS || Decide the reference basis
    i=0
    iref=0
    dv_sim=0
    SAME = False
    for ibas in TRbasis:       
       if ibas.upper() == basis.upper(): 
          iref = i
          dv_sim = 1
          SAME = True
          break
       else:
          dv=Similarity.basis(ibas,basis,mol) 
       if dv > dv_sim :
          iref = i
          dv_sim = dv
       i=i+1   

    if SAME:
       print("Identical reference basis space (",TRbasis[iref],"), detected. ")
    else:   
       print("Automaticly selected reference basis is ", TRbasis[iref], "with similarity value : ", dv_sim)

    ref_basis=TRbasis[iref]


    # || FUNCTIONAL || Decide the reference functional
    iladder=JacobLadder.position(funct)
    ref_funct=JacobLadder.typicalfunct(iladder)

    i=0
    iref=0
    SAME = False
    for ifun in TRfunct:
       if ifun.upper() == funct.upper():
          iref = i
          SAME = True
          ref_funct=ifun
          break

    if SAME:  
       print("Identical reference DFT functional space (",ref_funct,"), detected. ") 
    else: 
       print("Automaticly selected reference DFT functional is ",ref_funct, " from Jacob's ladder.")

    return ref_funct, ref_basis

