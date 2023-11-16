# Initial version written by Zhiying Li, re-written/updated by Yingjin

import os
import sys
import re
from operator import attrgetter
import math
from LBlibrary import *

nnode = 50

#dataR = "../example/test30_20%.pdb_GAU_DFT"
dataX = "../P38_LSTM_631gss2_augV"
#dataX = "../Predicted_LSTM_Solvent"

# Practical usage
outfile = "LB-LSTM" + "_P38_" + str(nnode) + "_631gss2_augV"
#outfile = "LB-LSTM" + "_SLV2_" + str(nnode)
write_outfile = True

frags = []
print(dataX)
print("nodes: " + str(nnode))

dxdx=readDataPRE(dataX)

#print(dxdx)
#exit(0)
#print(tdx)

dxdy=sorted(dxdx,key=lambda dxdx:dxdx[0])

# ==============================================
# in order to match julia order 
dxdy2=[]
for i in range(len(dxdy)-1):
    #print(i) 
    if i%2 == 0 :
        dxdy2.append(dxdy[i+1])
    else : 
        dxdy2.append(dxdy[i-1])
dxdy2.append(dxdy[len(dxdy)-1])

dxdy=[]
dxdy=dxdy2

#print(dxdy)
#exit(0)
# ==============================================
# in order to match julia order 

#print(len(dxdy))



pair2frags(dxdy,frags)




#print(frags)
#exit(0)
#readData3(dataX, frags)
print(len(frags))
#exit(0)
#ideal(frags, nnode, outfile, write_outfile)  # without multi-nodes





ideal2g(frags, nnode, outfile, write_outfile) #    with multi-nodes 





# readData(dataR, frags)
# ideal(frags, nnode, outfile, write_outfile)
# alphabetical(frags, nnode)
# bigFirst(frags, nnode, '../test20-pdb')

# Theorical analysis
# readData2(dataR, dataX, frags)
# print(len(frags))
# ideal(frags, nnode, outfile, write_outfile)
# alphabetical(frags, nnode)
# bigFirst(frags, nnode, '../test30-pdb', outfile, write_outfile)

# Practical usage
#readData3(dataX, frags)
#print(len(frags))
# bigFirst(frags, nnode, '../LBtest29-6-80A_para-pdb', outfile,
#          write_outfile)  #生成bigfirst解决方案
# ideal(frags, nnode, outfile, write_outfile) yy #生成greedy解决方案
#ideal2(frags, nnode, outfile, write_outfile)  #生成带跨节点的解决方案

# reduced
# readData2(dataR, dataX, frags)
# print(len(frags))
# reduced(frags,dataR)

# MRE and MAE
# readData2(dataR, dataX, frags)
# print(len(frags))
# compute(frags)

