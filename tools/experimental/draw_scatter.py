#PREFIX="C:/codes/Fcst_sys/"
PREFIX="/home/yingjin/Softwares/Fcst_sys/"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


ground=[[] for i in range(4)]
exited=[[] for i in range(4)]

modlist=['rf','lstm','mpnn','mgcn']
modlist_2=['RF','LSTM','MPNN','MGCN']

funcs=['M062x']
bases=['6-31g','6-31gs','6-31pgs']
basis_dic={'6-31g':0,'6-31gs':1,'6-31pgs':2}
bases_2=['6-31G','6-31G*','6-31+G*']


#read data
for func in funcs:
    for basis in bases:
        chemspace=func+'_'+basis
        for i in range(len(modlist)):
            mod=modlist[i]
            with open(PREFIX+'model_all/'+mod+'_'+chemspace+'.log','r') as f:
                lines=f.readlines()
                for line in lines:
                    aline=line.strip().split(' ')
                    entry=[]
                    entry.append(basis)
                    for j in range(1,len(aline)):
                        entry.append(float(aline[j])) 
                    ground[i].append(entry)
            with open(PREFIX+'model_all/'+'TD_'+mod+'_'+chemspace+'.log','r') as f:
                lines=f.readlines()
                for line in lines:
                    aline=line.strip().split(' ')
                    entry=[]
                    entry.append(basis)
                    #entry.append(float(aline[j]) for j in range(1,len(aline)) )
                    for j in range(1,len(aline)):
                        entry.append(float(aline[j]))
                    exited[i].append(entry)


print(len(ground[1]))
color=['aquamarine','greenyellow','lightskyblue','salmon']
patches = [mpatches.Patch(color=color[i], label="{:s}".format(modlist_2[i])) for i in range(len(color))]
fig=plt.figure()
ax=[]

for i in range(2):
    ax.append(fig.add_subplot(1,2,i+1,projection='3d'))
    plt.xticks(np.arange(0,len(bases)),bases_2,rotation=45,fontsize=15)
    plt.yticks([])

for i in range(2):
    data=ground
    title='Ground'
    if i==1:
        data=exited
        title='TDDFT'
    ax[i].set_zlim(-1.0,1.0)
    #ax[i].set_title(title)
#    ax[i].legend(loc='upper left', fontsize=10,handles=patches)
    ax[i].legend(loc='upper right',fontsize=15,handles=patches)
    for j in range(len(modlist)):
        bas=[data[j][x][0] for x in range(len(data[j]))]
        locbas=[basis_dic[bas[x]]+0.17*j for x in range(len(bas))]
        bnum=[data[j][x][1] for x in range(len(data[j]))]
        re=[data[j][x][4] for x in range(len(data[j]))]
        #print(len(re[j]))
        ax[i].scatter(locbas,bnum,re,color=color[j])
#        ax[i].set_zlabel('   MRE')
        
plt.show()


