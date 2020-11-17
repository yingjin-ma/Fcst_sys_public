import os
import sys
import re

sdf_input='./411763242036096740.sdf'
smiles_input='./3194848463444947528.txt' # Can also be blank '' if no redundant
sdf_folder='./sdf_folder'

# Generate the recording folder
target_folder=sdf_folder
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# Only record the valid SMILES
nsmiles=[]
with open(smiles_input,'r') as smiles:
    lines = smiles.readlines()
    for line in lines:
        if len(line.split()) > 1:
            nsmiles.append(line.split()[0])

# Split the whole sdfs
nsdf=[]
with open(sdf_input,'r') as sdf_one:
    lines = sdf_one.readlines()
    iline=0
    for line in lines:
        iline=iline+1
        if len(line.split()) != 0:
            #print('line : ',line) 
            if line.split()[0] == '$$$$' :
                #print('match $$$$ : ',line) 
                nsdf.append(iline)

# Generate the valid sdfs
#print(len(nsdf))
with open(sdf_input,'r') as sdf_one:
    lines = sdf_one.readlines()
    iline=0
    csdf=lines[0].split()[0]
    if csdf in nsmiles:
        sdf_output=target_folder+'/'+csdf+'.sdf'
        with open(sdf_output,'w') as fsdf:
            for j in range(0,int(nsdf[0])):
                fsdf.write(lines[j])
    for isdf in range(len(nsdf)-2):
        #print("nsdf[isdf] : ",nsdf[isdf])
        csdf=lines[int(nsdf[isdf])].split()[0]
        #print("lines : ",lines[int(nsdf[isdf])])
        if csdf in nsmiles:
            sdf_output=target_folder+'/'+csdf+'.sdf'
            with open(sdf_output,'w') as fsdf:
                for j in range(int(nsdf[isdf]),int(nsdf[isdf+1])):
                    fsdf.write(lines[j])

#print(nsmiles)


