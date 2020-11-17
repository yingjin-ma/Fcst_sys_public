import os
import re

#PREFIX="C:/codes/Fcst_sys/"
PREFIX="/home/yingjin/Softwares/Fcst_sys/"

models=['Lstm','MGCN','MPNN']
# tb_list=["SV","SVP","cc-PVDZ","cc-PVTZ","6-31ppgss"]
# ob_list=['631g','631gs','631pgs']

basis_dict={'SV':['631g'],'SVP':['631gs'],'cc-PVDZ':['631gs'],
'cc-PVTZ':['631gs','631pgs'],'6-31ppgss':['631pgs']}

td=['notrained2','bothnotrained2']
data=['20200728','20200824']

rootDir="logData-figs/"


def rename(src):
    if src=='SV' or src=='SVP':
        return src
    sdic={'cc-PVDZ':'cc-pVDZ','cc-PVTZ':'cc-pVTZ','6-31ppgss':'6-31ppGss'}
    return sdic[src]

def extract():
    if not os.path.exists(PREFIX+rootDir):
        os.mkdir(rootDir)
    for model in models:
        for tb in basis_dict.keys():
            for ob in basis_dict[tb]:
                for i in range(len(data)):
                    slog="log.run_g09_Fcst_"+model+"-"+td[i]+"-"+ob+"-"+data[i]
                    slogLoc=PREFIX+"Spaces/logs_Figs/"+slog
                    entries=[]
                    #with open(slogLoc,"r",encoding="utf8") as f:
                    with open(slogLoc,"r") as f:
                        for line in f.readlines():
                            #print("M06-2x_"+rename(tb))
                            if re.match(r"i:  ",line) and line.find("M06-2x_"+rename(tb))>-1:
                                sdf=re.findall(r'sdf/mol:  (\d+_[a-zA-Z0-9-]+_[a-zA-Z0-9-]+)',line)
                                #print(sdf)
                                if len(sdf)>0:
                                    res=re.findall(r'[-]?\d+\.\d+',line)
                                    if len(res)<4:
                                        continue
                                    res[3]=str((float(res[2])-float(res[1]))/float(res[1]))
                                    sdf.extend(res)
                                    #print(sdf)
                                    entries.append(" ".join(sdf)+"\n") 
                    tfile=PREFIX+"tools/"+rootDir+model.upper()+"_"+"M062x_"+tb+"_"+ob+"_"+td[i]+".log"
                    with open(tfile,"w") as target:
                        target.writelines(entries)

extract()

# def extract2():
#     if not os.path.exists(PREFIX+rootDir):
#         os.mkdir(rootDir)
#     logList=['log.run_g09_Fcst_Lstm-notrained-lcblyp',
#     'log.run_g09_Fcst_MPNN-notrained-lcblyp2',
#     'log.run_g09_Fcst_MGCN-notrained-lcblyp2']
#     for name in logList:
#         loc=PREFIX+'Spaces/'+name
#         with open(loc,'r',encoding="utf8") as logFile:


