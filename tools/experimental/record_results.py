import os
import sys
#import time
import re
from os import path

cdir=path.dirname('.')
apath=path.abspath(cdir)
dpath=apath.split('/')
comb=dpath[len(dpath)-1]

def time_gau(gjf):
    glog=gjf.split('.')[0]+'.log'
    name=glog.split('.')[0]
    cpu_time=[] # contains 4 elements:days,hours,minutes,seconds
    elp_time=[] # same as cpu_time
    nbasis=0
    with open(glog,'r') as log:
        for line in log:
            imatch_time1=re.match(" Job cpu time",line)
            imatch_time2=re.match(" Elapsed time",line)
            if imatch_time1 != None :
                for i in range(3,10,2):
                    cpu_time.append(float(line.split()[i]))
            if imatch_time2 != None :
                for i in range(2,9,2):
                    elp_time.append(float(line.split()[i]))
            imatch_basis=re.match(" There are",line)
            if imatch_basis !=None:
                try:
                    nbasis=int(line.split()[2]) #基函数数目
                except Exception:
                    return None
    print("cpu_time : ",cpu_time)  
    print("elp_time : ",elp_time)
    if len(elp_time)<4: 
        for i in range(0,4,1):   
            elp_time.append(0.0)
#    if len(cpu_time)<4 or len(elp_time)<4:
#        return None
    time_cpu=24*3600*cpu_time[0]+3600*cpu_time[1]+60*cpu_time[2]+cpu_time[3]
    time_elp=24*3600*elp_time[0]+3600*elp_time[1]+60*elp_time[2]+elp_time[3]

    print("time_cpu : ", time_cpu)
    print("time_elp : ", time_elp)

    return [nbasis,0,time_cpu,time_elp,name]


def write_res(res):
    if res==None or len(res)<5:
        return
    with open(comb+'.txt','a+') as f:
        tmp=len(res)-1
        for i in range(tmp):
            f.write(str(res[i])+" ")
        f.write(str(res[tmp]))
        f.write('\n')



mols=[]
moldir="./"
for root,dirs,files in os.walk(moldir):
    for f in files:
        if f.split('.')[1]=='log':
            mols.append(str(f))

for m in mols:
    res=time_gau(m)
    print(res)
    write_res(res)

