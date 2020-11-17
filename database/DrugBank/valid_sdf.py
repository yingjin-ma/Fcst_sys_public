import os
import sys
import re

sdf_folder='./sdf_folder'
sdf_folder_reduced='./sdf_only_valid_atoms'

valid_atoms   = ['H','He','Li','Be','B','C','N','O','F','Ne']
invalid_atoms = []

# Generate the recording folder
target_folder=sdf_folder_reduced
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# Only record the sdfs using the valid atoms
nsdf_list=os.listdir(sdf_folder)

for sdf in nsdf_list:
    csdf=sdf_folder+'/'+sdf
    ivalid=1
    with open(csdf,'r') as fsdf:
        lines = fsdf.readlines()
        for line in lines:
            if len(line.split()) > 15:
                if line.split()[3] not in valid_atoms:
                    ivalid=-1
                    break
    if ivalid == 1:
        oper='cp '+csdf+' '+target_folder
        os.system(oper)



#print(nsdf_list)


