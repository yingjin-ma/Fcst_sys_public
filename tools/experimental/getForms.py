import matplotlib.pyplot as plt
import numpy as np

prefix='C:/codes/Fcst_sys/'


models=['rf','lstm','mpnn','mgcn']
# path0=prefix+'Spaces_others/rf-res.log'
# path1=prefix+'Spaces_others/lstm-res.log'
# path2=prefix+'Spaces_others/mpnn-res.log'
# path3=prefix+'Spaces_others/mgcn-res.log'

dic=dict()

for m in models:
    dic[m]=dict()
    with open(prefix+'Ground_states_MOLCAS/'+m+'-res-tot.log','r') as f:
        lines=f.readlines()
        for line in lines:
            func_basis=line.split(' ')[0]
            func=func_basis.split('_')[0]
            basis=func_basis.split('_')[1]
            mre=float(line.split(' ')[3].split(',')[0])
            var=float(line.split(' ')[7])
            if func not in dic[m].keys():
                dic[m][func]=dict()
            if basis not in dic[m][func].keys():
                dic[m][func][basis]=dict()
            dic[m][func][basis]=[mre,var]

#dic['mpnn']['CAM-B3LYP']['6-31pgs'][0]=0.5048116895232875

mres=dict()
for m in models:
    mres[m]=dict()
    for b in dic[m]['BLYP'].keys():
        if b not in mres[m].keys():
            mres[m][b]=[]
        for func in dic[m].keys():
            mres[m][b].append(dic[m][func][b][0])
            


#fig,subs=plt.subplots(2,2)
plt.figure()
bar_width=0.35
x=range(len(dic['lstm'].keys()))
colors=['orangered','pink','dodgerblue']

for fi in range(4):
    ax=plt.subplot(221+fi)
    j=0
    for b in dic['rf']['BLYP'].keys():
        ax.bar([i-0.2+j*0.2 for i in x],height=mres[models[fi]][b],width=0.2,color=colors[j],alpha=0.8,label=b)
        j+=1
    
    ax.set_title(models[fi])
    ax.set_xticks(x)
    ax.set_xticklabels(dic['lstm'].keys(),rotation=30,fontsize=7)
    #ax.set_xlabel('functionals')
    ax.set_ylabel('mre')
    ax.legend()



plt.show()