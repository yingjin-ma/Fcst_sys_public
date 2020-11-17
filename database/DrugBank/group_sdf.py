import os
import sys
import re
import numpy as np
import random
#from rdkit import Chem

# Step-1 : for groupping sdfs
sdf_folder     = './sdf_only_valid_atoms'
sdf_folder2    = './sdf_only_valid_atoms_working' # recording valid sdfs for the later training
interval       =   2         # every N atoms
aiming_atoms   = ['Li','Be','B','C','N','O','F','Ne']   # atoms that used for grouping
ngroup         =  50         # record into N groups
group_folder   = 'SRow_grouped' # recording folder of groups
group_start    =    1
group_finish   =   25 

# Step-2 : for genearting training/validing/tesing suits
neach  = [5,1,1]  # Number of training/validing/tesing sdf in each group
folder = 'suits'

# Only record the sdfs using the valid atoms
nsdf_list=os.listdir(sdf_folder)

oper='rm -rf '+group_folder+'*'
os.system(oper)

nsdf=len(nsdf_list)
sdf_num=np.zeros(ngroup+1)
for sdf in nsdf_list:
    csdf=sdf_folder+'/'+sdf
    iatom=0
    with open(csdf,'r') as fsdf:
        lines = fsdf.readlines()
        for line in lines:
            if len(line.split()) > 15:
                if line.split()[3] in aiming_atoms:
                    iatom=iatom+1

#    print(iatom)
    if iatom/interval < ngroup:
        sdf_num[int(iatom/interval)]=sdf_num[int(iatom/interval)]+1
        igroup=int(iatom/interval)
    else:
        sdf_num[ngroup]=sdf_num[ngroup]+1
        igroup=ngroup

    if (igroup >= group_start) and (igroup <= group_finish):
       cfolder=group_folder+'-'+str(igroup)
       # Distribute into recording group folder
       target_folder=cfolder
       if not os.path.exists(target_folder):
          os.mkdir(target_folder)

       oper='cp '+csdf+' '+target_folder
       os.system(oper)

for i in range(ngroup):
    if (i >= group_start) and (i <= group_finish):
      print('==> ', int(sdf_num[i]),' SDFs in ' + group_folder +'-'+str(i)+' folder',' with aiming atoms ',aiming_atoms,' : '+str(i*2)+'-'+str((i+1)*2-1))
#print('==> ', int(sdf_num[ngroup]),' SDFs in ' + group_folder+'-'+str(ngroup)+' (extra, not used) folder'+' with aiming atoms ',aiming_atoms,' > '+str((ngroup)*2))

#print(sdf_num)

cgroups=[]
cdirs=os.listdir('./')
for cgroup in cdirs:
    if cgroup.startswith(group_folder):
        cgroups.append(cgroup)

#print(cgroups)

oper='rm -rf '+folder + '-*'
os.system(oper)
igroup=0
for cgroup in cgroups:
    igroup=igroup+1
    group_list=os.listdir(cgroup)
    nlist=len(group_list)
    # random number for shifting the position for sdf
    #ishift=random.randint(0,nlist)
    ishift=0
    #print(nlist)
    #print(group_list)

    if igroup == ngroup:
        break

    cfolder=folder + '-training'
    target_folder=cfolder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cfolder=folder + '-validing'
    target_folder=cfolder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    cfolder=folder + '-testing'
    target_folder=cfolder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    # training suit
    for i in range(neach[0]):
        ii=ishift+i
        if i > nlist-1:
            break
        #print('training ii-0: ',ii)
        if ii > nlist-1:
            ii=ii-nlist
        #print('training ii-1: ',ii)
        oper = 'cp ' + cgroup + '/' + group_list[ii] + ' ' + folder + '-training'
        os.system(oper)
    # validing suit
    for i in range(neach[1]):
        ii=ishift+i+neach[0]
        if i+neach[0] > nlist-1:
            break
        #print('validing ii-0: ',ii)
        if ii > nlist-1:
            ii=ii-nlist
        #print('validing ii-1: ',ii)
        oper = 'cp ' + cgroup + '/' + group_list[ii] + ' ' + folder + '-validing'
        os.system(oper)
    # testing  suit
    for i in range(neach[2]):
        ii=ishift+i+neach[0]+neach[1]
        if i+neach[0]+neach[1] > nlist-1:
            break
        #print('testing  ii-0: ',ii)
        if ii > nlist-1:
            ii=ii-nlist
        #print('testing  ii-1: ',ii)
        oper = 'cp ' + cgroup + '/' + group_list[ii] + ' ' + folder + '-testing'
        os.system(oper)

print('The training suit (', folder+'-training)',' contains ', len(os.listdir(folder+'-training')),' sdfs')
print('The validing suit (', folder+'-validing)',' contains ', len(os.listdir(folder+'-validing')),' sdfs')
print('The testing  suit (', folder+'-testing )',' contains ', len(os.listdir(folder+'-testing')),' sdfs')

target_folder=sdf_folder2
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

oper = 'cp -r ' + folder+'-training/* ' +  sdf_folder2
os.system(oper)
oper = 'cp -r ' + folder+'-validing/* ' +  sdf_folder2
os.system(oper)
oper = 'cp -r ' + folder+'-testing/* ' +  sdf_folder2
os.system(oper)

#print(sdf_group_list)


