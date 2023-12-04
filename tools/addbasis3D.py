import os
import json
import socket

from rdkit import Chem
from rdkit.Chem import AllChem
import basis_set_exchange as bse

folder0 = "../database/rawdata/G09data.01/M062x_6-31pgs"
folder1 = "../database/rawdata/G09data.3D/M062x_6-31pgs"
sdf0    = "../database/rawdata/Arxiv1911.05569v1_sdfs_H"

lists=[]
for root,dirs,files in os.walk(folder0):
    for f in files:
        lists.append(str(f))

print(lists)
if not os.path.exists(folder1):
    os.mkdir(folder1)


bas="6-31pgs"

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


MOLbasis3D1=[]
MOLbasis3D2=[]
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
                    basvec1=[]
                    basvec2=[] 
                    for line1 in lines1:
                        i0=i0+1
                        if i0>4 :
                            if len(line1.split())>15:
                                naos3=0 
                                naop3=0 
                                naod3=0 
                                naof3=0 
                                naog3=0 
                                naoh3=0 
                                naos4=0 
                                naop4=0 
                                naod4=0 
                                naof4=0 
                                naog4=0 
                                naoh4=0 
                                #print(line1.split()[3])
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
                                        naos3 = int(ao1[n][:-1])
                                        naos4 = int(ao2[n][:-1])
                                    if ao2[n][1] == 'p':
                                        naop1 = naop1 +3*int(ao1[n][:-1])
                                        naop2 = naop2 +3*int(ao2[n][:-1])
                                        naop3 = 3*int(ao1[n][:-1])
                                        naop4 = 3*int(ao2[n][:-1])
                                    if ao2[n][1] == 'd':
                                        if D56 == -1:
                                            naod1= naod1+ 6*int(ao1[n][:-1])
                                            naod2= naod2+ 6*int(ao2[n][:-1])
                                            naod3= 6*int(ao1[n][:-1])
                                            naod4= 6*int(ao2[n][:-1])
                                        elif D56 == 1:
                                            naod1= naod1+5*int(ao1[n][:-1])
                                            naod2= naod2+5*int(ao2[n][:-1])
                                            naod3=5*int(ao1[n][:-1])
                                            naod4=5*int(ao2[n][:-1])
                                    if ao2[n][1] == 'f':
                                        naof1=naof1+ 7*int(ao1[n][:-1])
                                        naof2=naof2+ 7*int(ao2[n][:-1])
                                        naof3= 7*int(ao1[n][:-1])
                                        naof4= 7*int(ao2[n][:-1])
                                    if ao2[n][1] == 'g':
                                        naog1= naog1+9*int(ao1[n][:-1])
                                        naog2= naog2+9*int(ao2[n][:-1])
                                        naog3= 9*int(ao1[n][:-1])
                                        naog4= 9*int(ao2[n][:-1])
                                    if ao2[n][1] == 'h':
                                        naoh1=naoh1+11*int(ao1[n][:-1])
                                        naoh2=naoh2+11*int(ao2[n][:-1])
                                        naoh3=11*int(ao1[n][:-1])
                                        naoh4=11*int(ao2[n][:-1])
                                
                                bas3=[naos3,naop3,naod3,naof3,naog3,naoh3]
                                bas4=[naos4,naop4,naod4,naof4,naog4,naoh4]
                                


                                #print("bas3 : ", bas3) 
                                #print("bas4 : ", bas4) 
                     
                            basvec1.append(bas3) 
                            basvec2.append(bas4) 

                print("uncontracted : ",naos1,naop1,naod1,naof1,naog1,naoh1)
                print("  contracted : ",naos2,naop2,naod2,naof2,naog2,naoh2)

                MOLbasis3D1.append(basvec1)            
                MOLbasis3D2.append(basvec2)            

                lineupdated=line.replace("\n", "") + " [" + "uncontracted : " + str(naos1) + " " + str(naop1) + " " +str(naod1) + " " +str(naof1) +" " + str(naog1) + " " +str(naoh1) + "]" + " [" + "  contracted : " + str(naos2) +" " + str(naop2) + " " +str(naod2) + " " +str(naof2) + " " +str(naog2) + " " +str(naoh2) + "]"
                data2.write(lineupdated) 

                data2.write(" uncontracted_per : ")
                data2.write(str(basvec1))
                data2.write(" contracted_per : ") 
                data2.write(str(basvec2))
                data2.write("\n") 



#            print(risdf)

#    for inp in ifile:



