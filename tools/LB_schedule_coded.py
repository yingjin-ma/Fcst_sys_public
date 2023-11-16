import sys
import re
from operator import attrgetter
import math
from LBlibrary import *
import numpy as np

nnode = 50
# redundant_degree < nnode
redundant_degree = 1
#dataR = "../example/test30_20%.pdb_GAU_DFT"
dataX = "../Predicted_P38_LSTM_M062x-631gss"
#dataX = "../Predicted_LSTM_Solvent"

# Practical usage
outfile = "LB-LSTM" + "_P38_" + str(nnode)
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

pair2frags(dxdy,frags)


Rfrags, frags = get_Redundant_frags(frags)
#print(frags)
#exit(0)
#readData3(dataX, frags)
#print(len(frags))
#exit(0)
#ideal(frags, nnode, outfile, write_outfile)  # without multi-nodes
# ideal2g(frags, nnode, outfile, write_outfile)
ideal2g_coded(frags,Rfrags,nnode,redundant_degree, outfile, write_outfile)

















