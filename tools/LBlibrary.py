# Initial version written by Zhiying Li, re-written/updated by Yingjin

import os
import sys
import re
from operator import attrgetter
import math

#os.chdir(sys.path[0])

class puple:
    def __init__(self, nbas, telp, tcpu, mon1, mon2, fname):
        self.nbas = nbas
        self.telp = telp
        self.tcpu = tcpu
        self.mon1 = mon1
        self.mon2 = mon2
        self.fname = fname
        self.weight = 0


def extract_info(path):
    newPath = '../extracted/' + path.split('/')[2]
    with open(newPath, 'w') as fw:
        with open(path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                newline = line.split()[4] + " " + line.split()[2]
                fw.write(newline + "\n")

# 
def readDataPRE(pathX):
    with open(pathX, 'r') as fdata:
        dxdx=[]
        #idx=[] 
        #tdx=[] 
        for line in fdata:
            # print( int(line.split()[1].split(".")[0]), float(line.split()[2]) )
            i1 = (line.split()[1].split(".")[0])
            d1 = float(line.split()[2])
            #idx.append(i1) 
            #tdx.append(d1) 
            dxdx.append([i1,d1])
    #print(dxdx)        
    return dxdx        

def pair2frags(dxdx,frags):
    for i in range(len(dxdx)):
        #print(dxdx[i][1], " : ", dxdx[i][0] )
        #frags.append(puple(0, 0, float(dxdx[i][1]), 0, 0, dxdx[i][0]))
        frags.append(puple(0, 0, float(dxdx[i][1]), 0, 0, str(i)))


# read the predicted time, do the scheduling 
def readData3(pathX, frags):
    with open(pathX, 'r') as fdata:
        for line in fdata:
            frags.append(puple(0, 0, float(line.split()[1]), 0, 0,
                      line.split()[0].split('/')[2].split('.')[0]))

# predicted timing file, real timing file 
# only predicted results will be put on the frags
def readData2(pathR, pathX, frags):
    filename = []
    with open(pathX, 'r') as fdata:
        for line in fdata:
            filename.append(line.split()[0].split("/")[2].split(".")[0])

    with open(pathR, 'r') as fdata:
        for line in fdata:
            mol = line.split()[4].split("_")[2]
            name = line.split()[4].split("_")[1] + "_" + mol
            # print(name)
            if name not in filename:
                continue

            if len(mol.split("-")) == 2:
                frags.append(
                    puple(int(line.split()[0]), float(line.split()[1]),
                          float(line.split()[2]), int(mol.split("-")[0]),
                          int(mol.split("-")[1]),
                          line.split()[4]))
            if len(mol.split("-")) == 1:
                frags.append(
                    puple(int(line.split()[0]), float(line.split()[1]),
                          float(line.split()[2]), int(mol.split("-")[0]),
                          int(0),
                          line.split()[4]))


# real the practical time, refined as readData2 
def readData(path, frags):
    with open(path, 'r') as fdata:
        for line in fdata:
            mol = line.split()[4].split("_")[2]
            if len(mol.split("-")) == 2:
                frags.append(
                    puple(int(line.split()[0]), float(line.split()[1]),
                          float(line.split()[2]), int(mol.split("-")[0]),
                          int(mol.split("-")[1]),
                          line.split()[4]))
            if len(mol.split("-")) == 1:
                frags.append(
                    puple(int(line.split()[0]), float(line.split()[1]),
                          float(line.split()[2]), int(mol.split("-")[0]),
                          int(0),
                          line.split()[4]))

# theorical analysis
def task_assignment(frags, nnode, assigns):
    # assigns = []
    for i in range(nnode):
        assigns.append([0, []])

    for ifrag in frags:
        assigns.sort()
        assigns[0][0] += ifrag.tcpu  # assigns[0][0] += ifrag.telp
        if ifrag.mon2 == 0:
            assigns[0][1].append("test30_monomer_" + str(ifrag.mon1))
        else:
            assigns[0][1].append("test30_dimer_" + str(ifrag.mon1) + "-" +
                                 str(ifrag.mon2))

    return assigns

# practical applications
def task_assignment2(frags, nnode, assigns):
    # assigns = []
    for i in range(nnode):
        assigns.append([0, []])

    for ifrag in frags:
        print("ifrag : ",ifrag) 
        assigns.sort()
        assigns[0][0] += ifrag.tcpu  # assigns[0][0] += ifrag.telp
        # assigns[0][1].append(ifrag.fname)  # frag-name
        assigns[0][1].append(ifrag.fname.split('-')[1])

    return assigns


# practical applications
def task_assignment3(frags, nnode, assigns):
    # assigns = []
    for i in range(nnode):
        assigns.append([0, []])

    for ifrag in frags:
        print("ifrag : ",ifrag)
        assigns.sort()
        assigns[0][0] += ifrag.tcpu  # assigns[0][0] += ifrag.telp
        # assigns[0][1].append(ifrag.fname)  # frag-name
        assigns[0][1].append(ifrag.fname)

    return assigns


# static scheduling
def ideal(frags, nnode, outfile, write_outfile):
    frags = sorted(frags, key=attrgetter("tcpu"), reverse=True)
    # for ifrag in frags:
    #     print("nbas : ", ifrag.nbas, " Telp : ",ifrag.telp," Tcpu : ", ifrag.tcpu, " mon1 : ", ifrag.mon1," mon2 : ", ifrag.mon2)
    assigns = []
    total_time = []
    # task_assignment(frags, nnode, assigns)
    task_assignment3(frags, nnode, assigns)
    for assign in assigns:
        total_time.append(assign[0])
        # print(assign)
    if write_outfile:
        write_loadbalance(outfile, assigns)

    print("==============================ideal==============================")
    print(total_time)
    print("end-time:\t" + str(max(total_time)))
    print("utilization:\t" +
          str(sum(total_time) / (max(total_time) * len(total_time))))
    return assigns


# scheduling with multi-nodes
def ideal2(frags, nnode, outfile, write_outfile):
    multinodes = {}
    rate = 0.7  # 假设任务跨节点的并行效率是70%

    while True:
        assigns = ideal(frags, nnode, outfile, False)  # 上一次规划结果
        utilization = get_utilization(assigns)
        if utilization > 0.9:
            print('do not need to cross nodes')
            break
        else:
            print('need to cross nodes')
            frags = sorted(frags, key=attrgetter("tcpu"), reverse=True)
            maxfrag = frags.pop(0)
            maxfrag.tcpu = maxfrag.tcpu / (rate * 2)
            frags.append(maxfrag)
            frags.append(maxfrag)
            if (maxfrag.fname in multinodes):
                multinodes[maxfrag.fname] += 1
            else:
                multinodes[maxfrag.fname] = 2

            # for ifrag in frags:
            #     print("fname : ", ifrag.fname, " Tcpu : ", ifrag.tcpu)

    # print('Done')
    # print("multinodes ---->   ", multinodes)
    if write_outfile:
        write_loadbalance(outfile, assigns)
        write_crossnodes(outfile, multinodes)




def get_utilization(assigns):
    total_time = []
    for assign in assigns:
        total_time.append(assign[0])
    utilization = sum(total_time) / (max(total_time) * len(total_time))
    return utilization



def alphabetical(frags, nnode):
    frags = sorted(frags, key=attrgetter("fname"))
    # for ifrag in frags:
    #     print("nbas : ", ifrag.nbas, " Telp : ", ifrag.telp, " Tcpu : ",
    #           ifrag.tcpu, " mon1 : ", ifrag.mon1, " mon2 : ", ifrag.mon2,
    #           " filename : ", ifrag.fname)
    assigns = []
    total_time = []
    for i in range(nnode):
        assigns.append([0, []])

    assignIdx = 0
    for ifrag in frags:
        assigns[assignIdx][0] += ifrag.tcpu
        assigns[assignIdx][1].append(ifrag.fname)
        assignIdx += 1
        if (assignIdx >= nnode):
            assignIdx = 0
    # print(assigns)
    for assign in assigns:
        total_time.append(assign[0])
    print(
        "==============================alphabetical=============================="
    )
    print(total_time)
    print("end-time:\t" + str(max(total_time)))
    print("utilization:\t" +
          str(sum(total_time) / (max(total_time) * len(total_time))))



def compute_weight(frags, path):
    elements = {
        'H': 1,
        'C': 12,
        'O': 16,
        'O1-': 16,
        'N': 14,
        'N1+': 14,
        'F': 19,
    }
    for ifrag in frags:
        weight = 0
        if len(ifrag.fname.split('_')) > 1:
            filename = ifrag.fname.split('_')[1] + '-' + ifrag.fname.split(
                '_')[2]
        else:
            filename = ifrag.fname
        with open(path + '/' + filename + '.pdb', 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                elem = line.split()[-1]
                weight += elements[elem]
        ifrag.weight = weight



def bigFirst(frags, nnode, path, outfile, write_outfile):
    compute_weight(frags, path)
    frags = sorted(frags, key=attrgetter("weight"), reverse=True)
    # for ifrag in frags:
    #     print("nbas : ", ifrag.nbas, " Telp : ", ifrag.telp, " Tcpu : ",
    #           ifrag.tcpu, " mon1 : ", ifrag.mon1, " mon2 : ", ifrag.mon2,
    #           " filename : ", ifrag.fname, " weight : ", ifrag.weight)
    assigns = []
    total_time = []
    for i in range(nnode):
        assigns.append([0, []])

    assignIdx = 0
    for ifrag in frags:
        assigns[assignIdx][0] += ifrag.tcpu
        assigns[assignIdx][1].append(ifrag.fname)
        assignIdx += 1
        if (assignIdx >= nnode):
            assignIdx = 0
    # print(assigns)
    for assign in assigns:
        total_time.append(assign[0])

    if write_outfile:
        write_loadbalance(outfile, assigns)

    print(
        "==============================bigFirst==============================")
    print(total_time)
    print("end-time:\t" + str(max(total_time)))
    print("utilization:\t" +
          str(sum(total_time) / (max(total_time) * len(total_time))))



def write_loadbalance(outfile, assigns):
    lengths = []
    for assign in assigns:
        lengths.append(len(assign[1]))
    maxlen = max(lengths)
    nodes = len(assigns)
    print(assigns)
    with open(outfile, 'a') as outf:
        outf.write('nodes ' + str(nodes) + '\n')
        for assign in assigns:
            assign[1] = list(assign[1] + ['0'] * (maxlen - len(assign[1])))
            # print(assign[1])
            line = ' '.join(assign[1])
            outf.write(line + '\n')
        outf.write('\n')
    print('maxlen=' + str(maxlen))


def write_crossnodes(outfile, multinodes):
    crossnodes = {}
    for frag, nnode in multinodes.items():
        fragIdx = frag.split('-')[1]
        if nnode in crossnodes:
            crossnodes[nnode].append(fragIdx)
        else:
            crossnodes[nnode] = [fragIdx]

    with open(outfile, 'a') as outf:
        for nnode, frags in crossnodes.items():
            outf.write('multinodes ' + str(nnode) + '\n')
            line = ' '.join(frags)
            outf.write(line + '\n')
            outf.write('\n')


# different variations inputs
def reduced(frags, pathR):
    frags = sorted(frags, key=attrgetter("tcpu"))
    # for ifrag in frags:
    #     print("nbas : ", ifrag.nbas, " Telp : ",ifrag.telp," Tcpu : ", ifrag.tcpu, " mon1 : ", ifrag.mon1," mon2 : ", ifrag.mon2)

    # with open("../loadbalance/sorted-cluster3", 'a') as outf:
    #     for ifrag in frags:
    #         outf.write(" Tcpu : "+ str(ifrag.tcpu) + '\n')

    hashlist = {}
    i = 0
    for ifrag in frags:
        interval = math.floor(ifrag.tcpu / 5)
        # print(interval)
        if interval not in hashlist:
            hashlist[interval] = ifrag.fname
            i += 1
    print(i)

    boundary = 600  # 1020,960,860,420              1485, 1320,	1180, 600
    step = 2  # 6,3,2,2                       5,3,2,2
    i = 0
    count = 0
    with open("../example/test30_20%.pdb_GAU_TDDFT", 'a') as fw:
        with open("../example/test30_MAX.pdb_GAU_TDDFT", 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                if i <= boundary and i % step == 0:
                    fw.write(line)
                    count += 1
                elif i > boundary:
                    fw.write(line)
                    count += 1
                else:
                    fw.write(h20)
                i += 1

    print(i)
    print(count)



# Compute MAE and MRE
def compute(frags):
    times = []
    for ifrag in frags:
        times.append(ifrag.tcpu)

    # print(times)
    # print(len(times))
    average = sum(times) / len(times)
    mre = sum([(x - average)**2 for x in times]) / len(times)
    mae = math.sqrt(mre)

    print('mre: ', mre)
    print('mae: ', mae)


