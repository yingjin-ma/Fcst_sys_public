# Initial version written by Zhiying Li, re-written/updated by Yingjin

import os
import sys
import re
from operator import attrgetter
import math
import LB_schedule_library

os.chdir(sys.path[0])


nnode = 12

#dataR = "../example/test30_20%.pdb_GAU_DFT"
dataX = ""

# Practical usage
outfile = "LBfile" + "_" + str(nnode)
write_outfile = True

frags = []
print(dataR)
print("nodes: " + str(nnode))

readData3(dataX, frags)
print(len(frags))
ideal(frags, nnode, outfile, write_outfile)  # without multi-nodes
#ideal2(frags, nnode, outfile, write_outfile) #    with multi-nodes 

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

