import os
import json
import socket

from rdkit import Chem
from rdkit.Chem import AllChem
import basis_set_exchange as bse

folder0 = "/home/molaaa/Desktop/Fcst_sys_public/database/rawdata/G09data.01/B3LYP_6-31g"
folder1 = "/home/molaaa/Desktop/Fcst_sys_public/database/rawdata/G09data.01.updated/B3LYP_6-31g"
sdf0    = "/home/molaaa/Desktop/Fcst_sys_public/database/rawdata/Arxiv1911.05569v1_sdfs_H"

lists=[]
for root,dirs,files in os.walk(folder0):
    for f in files:
        lists.append(str(f))

print(lists)


bas="6-31g"

D56=-1
if bas == "cc-pVDZ":
   D56=1
elif bas == "cc-pVTZ":
   D56=1
else:
   bas=bas.replace('p','+').replace('s','*')

if bas == "SVP":
   bas=bas.replace('SVP','SVP (Dunning-Hay)')
   D56=1
if bas == "SV":
   bas=bas.replace('SV','SV (Dunning-Hay)')

elemdict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'S': 16, 'Cl': 17}


for ilist in lists:
    
    ifile=folder0+"/"+ilist
#    print(ifile) 
    risdf2=folder1+"/"+ilist
    with open(risdf2,'w') as data2: 
        with open(ifile,'r') as data0:
            lines = data0.readlines()
            for line in lines:
#            print(line)
                isdf = line.split()[4]
                risdf= sdf0 + "/" + isdf+".sdf"
                print(risdf) 
#            suppl = Chem.SDMolSupplier(risdf)
#            for atom in suppl[0].GetAtoms():
#                print(atom.GetAtomicNum())
          
                naos1=0 
                naop1=0 
                naod1=0 
                naof1=0 
                naog1=0 
                naoh1=0 
                naos2=0 
                naop2=0 
                naod2=0 
                naof2=0 
                naog2=0 
                naoh2=0 
            #with open(risdf2,"a") as data2: 
                #data2.write(line)
                with open(risdf,"r") as data1:
                    lines1 = data1.readlines()
                    i0=0 
                    for line1 in lines1:
                        i0=i0+1
                        if i0>4 : 
                            if len(line1.split())>15:
                                #print(line1.split()[3])
                                import pdb
                                pdb.set_trace()
                                natom=elemdict[line1.split()[3]]
                                bs_str = bse.get_basis(bas, elements=[natom], fmt='nwchem', header=False)
                                #print(bs_str) 
                                ao1=bs_str.split()[7].strip('(').strip(')').split(',')
                                ao2=bs_str.split()[9].strip('[').strip(']').split(',')
                                #print("ao1",ao1) 
                                #print("ao2",ao2) 
                                n2=len(ao2)
                                for n in range(n2):
                                    if ao2[n][1] == 's':
                                        naos1 = naos1 +  int(ao1[n][:-1])
                                        naos2 = naos2 +  int(ao2[n][:-1])
                                    if ao2[n][1] == 'p':
                                        naop1 = naop1 +3*int(ao1[n][:-1])
                                        naop2 = naop2 +3*int(ao2[n][:-1])
                                    if ao2[n][1] == 'd':
                                        if D56 == -1:
                                            naod1= naod1+ 6*int(ao1[n][:-1])
                                            naod2= naod2+ 6*int(ao2[n][:-1])
                                        elif D56 == 1:
                                            naod1= naod1+5*int(ao1[n][:-1])
                                            naod2= naod2+5*int(ao2[n][:-1])
                                    if ao2[n][1] == 'f':
                                        naof1=naof1+ 7*int(ao1[n][:-1])
                                        naof2=naof2+ 7*int(ao2[n][:-1])
                                    if ao2[n][1] == 'g':
                                        naog1= naog1+9*int(ao1[n][:-1])
                                        naog2= naog2+9*int(ao2[n][:-1])
                                    if ao2[n][1] == 'h':
                                        naoh1=naoh1+11*int(ao1[n][:-1])
                                        naoh2=naoh2+11*int(ao2[n][:-1])
                           
                print("uncontracted : ",naos1,naop1,naod1,naof1,naog1,naoh1)
                print("  contracted : ",naos2,naop2,naod2,naof2,naog2,naoh2)

                lineupdated=line.replace("\n", "") + " [" + "uncontracted : " + str(naos1) + " " + str(naop1) + " " +str(naod1) + " " +str(naof1) +" " + str(naog1) + " " +str(naoh1) + "]" + " [" + "  contracted : " + str(naos2) +" " + str(naop2) + " " +str(naod2) + " " +str(naof2) + " " +str(naog2) + " " +str(naoh2) + "]\n"
                data2.write(lineupdated)    


#            print(risdf)

#    for inp in ifile:



