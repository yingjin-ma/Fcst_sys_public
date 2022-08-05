import sys
import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem

tarfile = "P38-3fmh"


def add_nitrogen_charges(smiles):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(m)
    return m

moldir="./test"

with open(tarfile, 'r') as fsmiles:
    lines = fsmiles.readlines()

    #print("lines : ", lines)    
    for line in lines:
        ismi  = line.split()[0]
        iname = line.split()[1]
        #rint("ismi : ", ismi )
        msmi = add_nitrogen_charges(ismi)
        ismi2= Chem.MolToSmiles(msmi)
        qmol  = Chem.MolFromSmiles(ismi2)

        #rint("qmol : ", qmol)
        Chem.AddHs(qmol)
        AllChem.EmbedMolecule(qmol)
        #outputT=moldir+"/"+"tmp.sdf"
        outputT=moldir+"/"+iname+".sdf"
        print(Chem.MolToMolBlock(qmol),file=open(outputT,'w+'))
        #qmol.UpdatePropertyCache(strict=False)
        #mols.append(qmol)

