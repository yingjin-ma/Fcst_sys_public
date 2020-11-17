import sys
import os
import time
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem

class puple:
    def __init__(self,dft,bas,mre,mae,std,mod):
        self.dft  =  dft
        self.bas  =  bas
        self.mre  =  mre
        self.mae  =  mae
        self.std  =  std
        self.mod  =  mod


dft_norms={'PBE':'PBE1PBE','M06-2x':'M062x'}

PREFIX="/home/yingjin/Softwares/Fcst_sys/"

locs=["Spaces_GAMESS-refined","Ground_states_NWChem","Ground_states_MOLCAS"]
nwFiles=["rf-res-tot.log","lstm-res-tot.log","mpnn-res-tot.log","mgcn-res-tot.log"]

modlist=["RF","LSTM","MPNN","MGCN"]
funcs=['B3LYP','bhandhlyp','BLYP','CAM-B3LYP','LC-BLYP','M06','M062x','PBE1PBE','wb97xd']
bases=['6-31g','6-31gs','6-31pgs']
bases2=['6-31G','6-31G*','6-31+G*']

# res_gamess=[dict() for i in range(len(modlist))]
# res_nwchem=[dict() for i in range(len(modlist))]
# res_molcas=[dict() for i in range(len(modlist))]


# read gaussian data

res_gau=[dict() for i in range(len(modlist))]
for imod in range(len(res_gau)):
    for d in funcs:
        res_gau[imod][d]=dict()
        for b in bases:
            res_gau[imod][d][b]=puple(d,b,0,0,0,modlist[imod])
i=0
with open(PREFIX+"/Spaces/resultsTAB2TOTAll",'r') as f:
    for line in f.readlines():
        cont_a=line.split("&        &")[0].split('&')[2:]
        cont_a='&'.join(cont_a)
        cont_b=line.split("&        &")[1].split('&')[2:]
        cont_b='&'.join(cont_b)
        
        #print(cont_a," ",cont_b)
        i+=1
        dft=line.split('&')[1].strip()
        if dft in dft_norms.keys():
            dft=dft_norms[dft]
        str_vals1=re.findall(r"\d+\.?\d*",cont_a)
        str_vals2=re.findall(r"\d+\.?\d*",cont_b)
        #print(str_vals1," ",str_vals2)
        if dft not in res_gau[0].keys():
            for x in range(4):
                res_gau[x][dft]=dict()
        if i<=8:
            for j in range(len(bases)):
                res_gau[0][dft][bases[j]]=puple(dft,bases[j],float(str_vals1[j*2]),
                0,float(str_vals1[j*2+1]),'RF')
                print(res_gau[0][dft][bases[j]].mre)
                res_gau[1][dft][bases[j]]=puple(dft,bases[j],float(str_vals2[j*2]),
                0,float(str_vals2[j*2+1]),'LSTM')
                print(res_gau[1][dft][bases[j]].mre)
        else:
            for j in range(len(bases)):
                res_gau[2][dft][bases[j]]=puple(dft,bases[j],float(str_vals1[j*2]),
                0,float(str_vals1[j*2+1]),'MPNN')
                print(res_gau[2][dft][bases[j]].mre)
                res_gau[3][dft][bases[j]]=puple(dft,bases[j],float(str_vals2[j*2]),
                0,float(str_vals2[j*2+1]),'MGCN')
                print(res_gau[3][dft][bases[j]].mre)




res=[]
for i in range(3):
    res.append([dict() for j in range(len(modlist))])


for idx in range(len(res)):
    reslist=res[idx]
    fileDir=PREFIX+locs[idx]+"/"
    for i in range(len(modlist)):
        for d in funcs:
            reslist[i][d]=dict()
            for b in bases: 
                reslist[i][d][b]=puple(d,b,0,0,0,modlist[i])
        filename=fileDir+nwFiles[i]
        with open(filename,'r') as f:
            for line in f.readlines():
                cont=line.split(' ')
                cont1=line.split('TOT')[1]
                df_bas=cont[0]
                basis=df_bas.split('_')[1]
                func=df_bas.split('_')[0]
                str_vals=re.findall(r"\d+\.?\d*",cont1)
                #print(str_vals)
                # if func not in reslist[i].keys():
                #     reslist[i][func]=dict()
                reslist[i][func][basis]=puple(func,basis,float(str_vals[0]),
                float(str_vals[1]),float(str_vals[2]),modlist[i])


# for x in range(len(res)):
#     for y in res[x]:
#         for elem in y:
#             print(elem.dft,elem.bas,elem.mre,elem.mae,elem.std,elem.mod)


fig=plt.figure()
ax=[]
#fig.tight_layout()
#plt.subplots_adjust(left=0.1,bottom=0.1,top=1,right=1,hspace=0.18,wspace=0)
for i in range(len(modlist)):
    ax.append(fig.add_subplot(2,2,i+1,projection='3d'))
    plt.xticks(np.arange(0,len(bases)),bases2,rotation=45,fontsize=8)
    plt.yticks(np.arange(0,5*(len(funcs)),5),funcs,rotation=0,fontsize=6) #泛函坐标


color=['dimgray','orangered','deepskyblue','lime']
ld_labels=['Gaussian','GAMESS','NWChem','OpenMolcas']
patches = [mpatches.Patch(color=color[i], label="{:s}".format(ld_labels[i])) for i in range(len(color))]


for i in range(len(modlist)):
    ax[i].legend(handles=patches,fontsize=8)
#    ax[i].set_title(modlist[i],loc='left')
    ax[i].set_zlim(0,1.0)
    for j in range(len(funcs)):
        dcolor1=float(j/len(funcs))
        dcolor2=1.0-float(j/len(funcs))
        dcolor3=random.random()
        gau_data=[res_gau[i][funcs[j]][b].mre for b in bases]
        gm_data=[res[0][i][funcs[j]][b].mre for b in bases]
        nw_data=[res[1][i][funcs[j]][b].mre for b in bases]
        mo_data=[res[2][i][funcs[j]][b].mre for b in bases]
        ax[i].bar(np.arange(len(bases))-0.2,gau_data,j*5,zdir='y',alpha=0.9,width=0.125,facecolor='dimgray',edgecolor = (dcolor1,dcolor2,dcolor3), label='Gaussian', lw=0.1)
        ax[i].bar(np.arange(len(bases))+0,gm_data,j*5,zdir='y',alpha=0.9,width=0.125,facecolor='orangered',edgecolor = (dcolor1,dcolor2,dcolor3), label='GAMESS', lw=0.1)
        ax[i].bar(np.arange(len(bases))+0.2,nw_data,j*5,zdir='y',alpha=0.9,width=0.125,facecolor='deepskyblue',edgecolor = (dcolor1,dcolor2,dcolor3), label='NWChem', lw=0.1)
        ax[i].bar(np.arange(len(bases))+0.4,mo_data,j*5,zdir='y',alpha=0.9,width=0.125,facecolor='lime',edgecolor = (dcolor1,dcolor2,dcolor3), label='MOLCAS', lw=0.1)
        
        

plt.show()
exit(0)
