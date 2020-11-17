#PREFIX="C:/codes/Fcst_sys/"

PREFIX="/home/yingjin/Softwares/Fcst_sys/"

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import numpy as np
from matplotlib.gridspec import GridSpec


class point:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z


basis_dict={'SV':['631g'],'SVP':['631gs'],'cc-pVDZ':['631gs'],
'cc-pVTZ':['631gs','631pgs'],'6-31ppgss':['631pgs']}

td=['notrained2','bothnotrained2']

rootDir="logData-figs/"

modlist=['LSTM','MPNN','MGCN']

funcs=['M062x']

basis_idx={'SV':0,'SVP':1,'cc-pVDZ':2,'cc-pVTZ (REF:6-31G*)':3,'cc-pVTZ (REF:6-31+G*)':4,'6-31++G**':5}

#color=['aquamarine','greenyellow','lightskyblue','salmon']
color=['greenyellow','lightskyblue','salmon']
patches = [mpatches.Patch(color=color[i], label="{:s}".format(modlist[i])) for i in range(len(modlist))]


def rename(src,ref):
    if src=='6-31ppgss':
        return '6-31++G**'
    elif src=='cc-pVTZ':
        if ref=='631gs':
            return 'cc-pVTZ (REF:6-31G*)'
        elif ref=='631pgs':
            return 'cc-pVTZ (REF:6-31+G*)'
    else:
        return src


def readData():
    res=[{},{}]
    for i in range(len(td)):
        t=td[i]
        for mod in modlist:
            res[i][mod]=[] #res[i]={"LSTM":[...],"MPNN":[...],"MGCN:[...]"}
            for func in funcs:
                for k in basis_dict.keys():
                    for b in basis_dict[k]:
                        loc=PREFIX+rootDir+mod+"_"+func+"_"+k+"_"+b+"_"+t+".log"
                        x_label=rename(k,b)
                        with open(loc,"r",encoding="utf8") as f:
                            for line in f.readlines():
                                bnum=float(line.split(" ")[1])
                                err=float(line.strip().split(" ")[4])
                                res[i][mod].append(point(basis_idx[x_label],bnum,err))
    return res
                            


def drawScatter():
    points=readData()
    fig=plt.figure()
#    gs=GridSpec(1,3)
    ax=[]
    for i in range(2):
#        ax.append(plt.subplot(gs[0, :2]))
        ax.append(fig.add_subplot(1,2,i+1,projection='3d'))
        plt.xticks(np.arange(0,len(basis_idx)),list(basis_idx.keys()),rotation=45,fontsize=12)
        plt.yticks([])
        
    for i in range(2):
        #x_major_locator=MultipleLocator(2)
        #title=td[i]
        ax[i].set_zlim(-1.0,1.0)
#        ax[i].set_xlim(-0.5,5.5)
        #ax[i].set_title(title)
        ax[i].legend(loc='upper right',fontsize=12,handles=patches)
        #ax[i].xaxis.set_major_locator(x_major_locator)
        for j in range(len(modlist)):
            data=points[1][modlist[j]]
            xs=[point.x for point in data]
            ys=[point.y for point in data]
            zs=[point.z for point in data]
            xs_loc=[xs[n]+0.17*j for n in range(len(xs))]
            ax[i].scatter(xs_loc,ys,zs,color=color[j],s=7)
    plt.show()

drawScatter()
