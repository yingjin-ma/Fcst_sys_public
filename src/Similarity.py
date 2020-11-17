#system
import json
import socket
import re

#rdkit : cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem

#basis 
import basis_set_exchange as bse

# rewrite base on Ma Shuo's similarity codes 

def sim_val(polyene_with_H,origin,target):

# Avoid the name error    
   D56=-1
   if origin == "cc-pVDZ":
      D56=1
   elif origin == "cc-pVTZ":
      D56=1
   else:
      origin= origin.replace('p','+').replace('s','*')

   if origin == "SVP":
      origin= origin.replace('SVP','SVP (Dunning-Hay)')
      D56=1
   if origin == "SV":
      origin= origin.replace('SV','SV (Dunning-Hay)')

   D56t=-1
   if target == "cc-pVDZ":
      D56t=1
   elif target == "cc-pVTZ":
      D56t=1
   else:
      target=target.replace('p','+').replace('s','*')

   if target == "SVP":
      target = target.replace('SVP','SVP (Dunning-Hay)')
      D56t=1
   if target == "SV":
      target = target.replace('SV','SV (Dunning-Hay)')

   basis1=origin
   basis2=target
   obshape={'s':1,'p':3,'d':6,'f':7,'g':9,'h':11}
   ob1_unContracted={'s':0,'p':0,'d':0,'f':0,'g':0,'h':0}
   ob2_unContracted={'s':0,'p':0,'d':0,'f':0,'g':0,'h':0}
   ob1_contracted={'s':0,'p':0,'d':0,'f':0,'g':0,'h':0}
   ob2_contracted={'s':0,'p':0,'d':0,'f':0,'g':0,'h':0}

   Npmv1=0
   Npmv2=0
   Ngto1=0
   Ngto2=0
   Nobt1=0
   Nobt2=0

   for atom in polyene_with_H.GetAtoms():
       atom_idx=atom.GetAtomicNum()
       bs_str1=bse.get_basis(basis1,elements=[atom_idx], fmt='NWChem', header=False)
       bs_str2=bse.get_basis(basis2,elements=[atom_idx],fmt='NWChem',header=False)
       bs_str1=bs_str1.split('\n')[1]
       bs_str2=bs_str2.split('\n')[1]
       pattern=re.compile(r'\(.*\)')
       bs_str1_a=re.findall(pattern,bs_str1)[0] #'(*s,*p,*d,...)'
       bs_str1_a=bs_str1_a.split('(')[1].split(')')[0].split(',')
       bs_str2_a=re.findall(pattern,bs_str2)[0]
       bs_str2_a=bs_str2_a.split('(')[1].split(')')[0].split(',')
       pattern=re.compile(r'\[.*\]')
       bs_str1_b=re.findall(pattern,bs_str1)[0] #'[*s,*p,*d,...]'
       bs_str1_b=bs_str1_b.split('[')[1].split(']')[0].split(',')#['*s','*p','*d',...]
       bs_str2_b=re.findall(pattern,bs_str2)[0]
       bs_str2_b=bs_str2_b.split('[')[1].split(']')[0].split(',')
       for item in bs_str1_a:
           num=int(item[:-1])
           ob1_unContracted[item[-1]]+=num
           Ngto1+=num*obshape[item[-1]]
       for item in bs_str1_b:
           num=int(item[:-1])
           if num==1:
               Npmv1+=obshape[item[-1]]
           ob1_contracted[item[-1]]+=num
           Nobt1+=num*obshape[item[-1]]
       for item in bs_str2_a:
           num=int(item[:-1])
           ob2_unContracted[item[-1]]+=num
           Ngto2+=num*obshape[item[-1]]
       for item in bs_str2_b:
           num=int(item[:-1])
           if num==1:
               Npmv2+=obshape[item[-1]]
           ob2_contracted[item[-1]]+=num
           Nobt2+=num*obshape[item[-1]]

   #calculate the Jaccard index
   r=min(ob1_contracted['s'],ob2_contracted['s'])
   +3*min(ob1_contracted['p'],ob2_contracted['p'])
   +6*min(ob1_contracted['d'],ob2_contracted['d'])
   +7*min(ob1_contracted['f'],ob2_contracted['f'])
   +9*min(ob1_contracted['g'],ob2_contracted['g'])
   +11*min(ob1_contracted['h'],ob2_contracted['h'])

   p=abs(ob1_contracted['s']-ob2_contracted['s'])
   +3*abs(ob1_contracted['p']-ob2_contracted['p'])
   +6*abs(ob1_contracted['d']-ob2_contracted['d'])
   +7*abs(ob1_contracted['f']-ob2_contracted['f'])
   +9*abs(ob1_contracted['g']-ob2_contracted['g'])
   +11*abs(ob1_contracted['h']-ob2_contracted['h'])
   jidx=r/(r+p)

   rho1=Npmv1/Ngto1
   rho2=Npmv2/Ngto2

   dvsim=jidx*Nobt1/Nobt2*(1-abs(rho1-rho2))

   return dvsim            


def basis(bas1,bas2,mol):

   dv=0
   dv=sim_val(mol,bas1,bas2)
#   print("Ref basis ", bas1, " Tar basis ", bas2, " with dv : ", dv)

   return dv 

