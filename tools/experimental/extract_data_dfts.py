import os
import re

#PREFIX="C:/codes/Fcst_sys/"
PREFIX="/home/yingjin/Softwares/Fcst_sys/"

models=['Lstm','MGCN','MPNN']
data=['trained-origin','notrained-pbe2','notrained-lcblyp2']
rootDir="logData-dfts/"
basis=['6-31G','6-31Gs','6-31pGs']


def extract():
    if not os.path.exists(PREFIX+rootDir):
        os.mkdir(rootDir)
    for model in models:
        for ibas in basis: 
           for i in range(len(data)):
              slog="log.run_g09_Fcst_"+model+"-"+data[i]
              slogLoc=PREFIX+"Spaces/data.draw_DFTs/"+slog
              entries=[]
              #with open(slogLoc,"r",encoding="utf8") as f:
              print(slogLoc)
              with open(slogLoc,"r") as f:
                 for line in f.readlines():
                    #print("M06-2x_"+rename(tb))
                    if re.match(r"i:  ",line) and line.find("M06-2x_"+ibas)>-1:
                       sdf=re.findall(r'sdf/mol:  (\d+_[a-zA-Z0-9-]+_[a-zA-Z0-9-]+)',line)
                       print(sdf)
                       if len(sdf)>0:
                          res=re.findall(r'[-]?\d+\.\d+',line)
                          if len(res)<4:
                             continue
                          res[3]=str((float(res[2])-float(res[1]))/float(res[1]))
                          sdf.extend(res)
                          print(sdf)
                          entries.append(" ".join(sdf)+"\n") 
              tfile=PREFIX+"tools/"+rootDir+model.upper()+"_"+"M062x_"+ibas+"_"+data[i]+".log"
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


