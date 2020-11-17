#PREFIX="C:/codes/Fcst_sys/"

PREFIX="/home/yingjin/Softwares/Fcst_sys/"

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import numpy as np


class point:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z

rootDir="logData-dfts/"

modlist=['LSTM','MPNN','MGCN']

funcs=['M062x']

basis=['6-31G','6-31Gs','6-31pGs']
basis2=[0,1,2]
data=['trained-origin','notrained-pbe2','notrained-lcblyp2']

#color=['aquamarine','greenyellow','lightskyblue','salmon']
color=['greenyellow','lightskyblue','salmon']
color2=['darkgrey','deepskyblue','darkviolet']
patches = [mpatches.Patch(color=color[i], label="{:s}".format(modlist[i])) for i in range(len(modlist))]

def readData():
   res=[{},{},{}]
   print(len(basis),len(modlist))
   for i in range(len(basis)):
      for mod in modlist:
         res[i][mod]=[] #res[i]={"LSTM":[...],"MPNN":[...],"MGCN:[...]"}
#         print(i,mod) 
         for func in funcs:
            ii=0
            for idat in data:  
               loc=PREFIX+rootDir+mod+"_"+func+"_"+basis[i]+"_"+idat+".log"
               print("loc",loc)
               ld=[] 
               with open(loc,"r") as f:
                  for line in f.readlines():
                     bnum=float(line.split(" ")[1])
                     err=float(line.strip().split(" ")[4])
                     ld.append(point(basis2[i],bnum,err))
               print(len(ld))      
               res[i][mod].append(ld)
               ii=ii+1
   return res                            


def drawScatter():
    points=readData()

#    print(len(points),len(points[0]),len(points[0][modlist[0]][0]))
#    exit(0)

    xidx=[0.00,0.05,0.10,0.25,0.30,0.35,0.50,0.55,0.60,1.00,1.05,1.10,1.25,1.30,1.35,1.50,1.55,1.60,2.00,2.05,2.10,2.25,2.30,2.35,2.50,2.55,2.60]
    xidx2=['Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP',
           'Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP',
           'Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP','Original','REF:PBE','REF:LC-BLYP' ]

    td=['M06-2x/6-31G','M06-2x/6-31G*','M06-2x/6-31+G*']

    fig=plt.figure()
    ax=[]
    for i in range(3):
       ax.append(fig.add_subplot(1,3,i+1,projection='3d'))
       plt.xticks(xidx,xidx2,rotation=45,fontsize=8)
       plt.yticks([])
        
    for i in range(3):
        #x_major_locator=MultipleLocator(2)
        title=td[i]
        ax[i].set_zlim(-1.0,1.0)
        ax[i].set_title(title,fontsize=10)
        ax[i].legend(loc='upper right',fontsize=12,handles=patches)
        #ax[i].xaxis.set_major_locator(x_major_locator)
        for j in range(len(modlist)):
           for k in range(len(data)):
              tmp=points[i][modlist[j]][k]

#              print(tmp, len(tmp), len(tmp))

              xs=[point.x for point in tmp]
              ys=[point.y for point in tmp]
              zs=[point.z for point in tmp]
              print(" ===> i,k,j <=== ",i,j,k)
              print(xs,ys,zs)
              xs_loc=[xs[n]+0.25*j+0.05*k for n in range(len(xs))]
              ax[i].scatter(xs_loc,ys,zs,color=color[j],linewidths=0.175,edgecolors=color2[k],s=7)
#              plt.scatter(xs_loc,ys,zs,color=color[j])
    plt.show()

drawScatter()
