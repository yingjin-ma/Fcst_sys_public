import os
import json
import socket

HOSTNAME = socket.gethostname()
PWD=os.getcwd()
SRC=PWD+"/src"
from rdkit import Chem
from rdkit.Chem import AllChem
import basis_set_exchange as bse

def getNbasis_noRDkit(bas="6-31g",sdf=""):

   elemdict = {'H': 1, 'Li': 3, 'Be': '4', 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 15, 'P': 16, 'S': 17, 'Cl': 18}

   print("sdf",sdf)
   D56=-1
   if bas == "cc-pVDZ":
      D56=1
   elif bas == "cc-pVTZ":
      D56=1
   else:
      bas=bas.replace('p','+').replace('s','*')

   if bas == "SVP":
      bas=bas.replace('SVP','SVP (Dunning-Hay)')
      D56=1
   if bas == "SV":
      bas=bas.replace('SV','SV (Dunning-Hay)')

   Nbasis=0
   naos=0

   atoms=[]
   with open(sdf,'r') as sdf_one:
      lines = sdf_one.readlines()
      print("lines[3] :",lines[3][0:3])
      natoms=int(lines[3][0:3])
      for i in range(natoms):
         i = i+1 
         atoms.append(lines[3+i].split()[3])

   #print(atoms)      

   for atom in atoms:
      natom=elemdict[atom] 
      bs_str = bse.get_basis(bas, elements=[natom], fmt='nwchem', header=False)
      ao=bs_str.split()[9].strip('[').strip(']').split(',')
      n2=len(ao)
      #print("bs_str : ",bs_str)
      #print("basis : ",ao)
      for n in range(n2):
         if ao[n][1] == 's':
            naos=naos+ int(ao[n][:-1])
         if ao[n][1] == 'p':
            naos=naos+ 3*int(ao[n][:-1])
         if ao[n][1] == 'd':
            if D56 == -1:
               naos=naos+ 6*int(ao[n][:-1])
            elif D56 == 1:
               naos=naos+ 5*int(ao[n][:-1])
         if ao[n][1] == 'f':
            naos=naos+ 7*int(ao[n][:-1])
         if ao[n][1] == 'g':
            naos=naos+ 9*int(ao[n][:-1])
         if ao[n][1] == 'h':
            naos=naos+11*int(ao[n][:-1])

   #print("sdf \n",sdf," \n naos \n",naos)
   Nbasis=naos

   return Nbasis


    

def getNbasis(bas="6-31g",sdf=""):

   print("sdf",sdf)
   D56=-1
   if bas == "cc-pVDZ":
      D56=1 
   elif bas == "cc-pVTZ":
      D56=1 
   else:
      bas=bas.replace('p','+').replace('s','*')

   if bas == "SVP":
      bas=bas.replace('SVP','SVP (Dunning-Hay)')
      D56=1
   if bas == "SV":   
      bas=bas.replace('SV','SV (Dunning-Hay)')

   Nbasis=0
   naos=0
   suppl=Chem.SDMolSupplier(sdf)
   #suppl.UpdatePropertyCache(strict=False)
   print("getNbasis, Chem.SDMolSupplier")
   mols = [x for x in suppl]
   mol=mols[0]
   AllChem.Compute2DCoords(mol)
   AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
   molH = Chem.AddHs(mol)
   print("getNbasis, Chem.AddHs")
   for atom in molH.GetAtoms():
      natom  = atom.GetAtomicNum()
      bs_str = bse.get_basis(bas, elements=[natom], fmt='nwchem', header=False)
      ao=bs_str.split()[9].strip('[').strip(']').split(',')
      n2=len(ao)
      #print("bs_str : ",bs_str)
      #print("basis : ",ao)
      for n in range(n2):
         if ao[n][1] == 's':
            naos=naos+ int(ao[n][:-1])
         if ao[n][1] == 'p':
            naos=naos+ 3*int(ao[n][:-1])
         if ao[n][1] == 'd':
            if D56 == -1: 
               naos=naos+ 6*int(ao[n][:-1])
            elif D56 == 1:  
               naos=naos+ 5*int(ao[n][:-1]) 
         if ao[n][1] == 'f':
            naos=naos+ 7*int(ao[n][:-1])
         if ao[n][1] == 'g':
            naos=naos+ 9*int(ao[n][:-1])
         if ao[n][1] == 'h':
            naos=naos+11*int(ao[n][:-1])

   #print("sdf \n",sdf," \n naos \n",naos)
   Nbasis=naos

   return Nbasis

def fitted_magns(basisnums,basisnums2,chemspace,ployfitted="test.log"):

   dv_magns=[]

   dft  =chemspace.split("_")[0]
   basis=chemspace.split("_")[1]

   a=0.0
   b=0.0
   c=0.0

   with open(ployfitted,'r') as fitted:
      for line in fitted:
         if line.split()[0] == dft:
            a=float(line.split()[1])
            b=float(line.split()[2])
            c=float(line.split()[3])
            break

   for x in range(len(basisnums)):
      dv1= a * float(basisnums[x])  * float(basisnums[x])  + b * float(basisnums[x])  + c 
      dv2= a * float(basisnums2[x]) * float(basisnums2[x]) + b * float(basisnums2[x]) + c
      dv_magns.append(float(dv2/dv1))

#   print(a,"x^2 + ",b," x + ",c)
#   exit(0)

   return dv_magns

def magnification(cs,cs0,aimming='tot',fitting='no'):

   dft=cs.split("_")[0]
   bas=cs.split("_")[1]
   dft0=cs0.split("_")[0]
   bas0=cs0.split("_")[1]

   dv=100.0
   # assuming datebase or libxc library
   # temporary use list instead
   list10=['PBE1PBE','BLYP']
   list11=['B3LYP','bhandhlyp']
   list12=['M06','M062x','CAM-B3LYP','LC-BLYP','wb97xd']

   DFTLIST=['PBE1PBE', 'B1B95', 'B1LYP', 'B3LYP', 'B3P86', 'B971', 'B972', 'B98', 'BB95', 'BHandH', 'BLYP', 'BMK', 'BP86', 'BPBE', 'BPW91', 'BTPSS', 'BV5LYP', 'BVP86', 'CAM-B3LYP', 'G96B95', 'G96P86', 'G96PBE', 'G96PW91', 'G96TPSS', 'G96V5LYP', 'G96VP86', 'HISSbPBE', 'HSEH1PBE', 'LC-BLYP', 'LC-wPBE', 'M05', 'M052X', 'M06', 'M06HF', 'O3LYP', 'OB95', 'OHSE1PBE', 'OHSE2PBE', 'OP86', 'OPBE', 'OPW91', 'OTPSS', 'OV5LYP', 'OVP86', 'PBE1PBE', 'PBEB95', 'PBEP86', 'PBEPW91', 'PBETPSS', 'PBEV5LYP', 'PBEVP86', 'PBEh1PBE', 'PBEhB95', 'PBEhP86', 'PBEhPBE', 'PBEhPW91', 'PBEhTPSS', 'PBEhVP86', 'PW91B95', 'PW91P86', 'PW91PBE', 'PW91PW91', 'PW91TPSS', 'PW91V5LYP', 'PW91VP86', 'SLYP', 'SPL', 'SV5LYP', 'SVP86', 'SVWN', 'SVWN5', 'TPSSh', 'X3LYP', 'XALYP', 'XAPL', 'XAV5LYP', 'XAVP86', 'XAVWN', 'XAVWN5', 'bhandhlyp', 'mPW1LYP', 'mPW1PBE', 'mPW1PW91', 'mPW3PBE', 'tHCTHhyb', 'wb97', 'wb97x', 'wb97xd','M062x']
   MAGNTOT=[1.0, 1.1586656864978766, 1.007359565307276, 1.0287085464617831, 0.9650675534485371, 1.0485916608511054, 1.0011103038314595, 1.0393862620197258, 1.1340245310710506, 1.0050358565208213, 1.0227760393575112, 1.1838246530653536, 0.9777977441489297, 1.0219338047990563, 1.0112885324044327, 1.1439562778803671, 1.0163544270537426, 0.976773999819376, 1.1500472888884747, 1.1571340620963273, 1.0038514858095686, 0.9933270021643666, 0.9812747513659458, 1.170378726254289, 1.0029802882172467, 1.0172817208612448, 1.103275302575271, 1.1270301596408216, 1.0649037529153977, 1.0495620085027404, 1.1366198692380651, 1.15870054068265, 1.1998461404224512, 1.2523751703278734, 0.9896013884602431, 1.1650792388033633, 1.077208053030064, 1.0575935637774603, 0.997375673048766, 1.0623534136241668, 0.9735158689011305, 1.1469811163736292, 1.0050138941412847, 1.0166825525731527, 1.0016693981928142, 1.1599341329338915, 1.01342378312864, 1.0412965603393902, 1.155319740781404, 0.9942948354973572, 0.9964034754587757, 1.0088076520675346, 1.1511232668565887, 0.9974190236263487, 1.008130086494108, 1.0389012049417337, 1.159418856343791, 0.9965110488591987, 1.1660462606782047, 0.9683114306874017, 0.9885157417989898, 0.9901654353155248, 1.1711366874376907, 1.0207702755372885, 1.0521865293096002, 1.0181130651397075, 0.9492434648252965, 0.9918033774578592, 1.014888738286248, 0.9481376784180297, 0.9435490976092861, 1.1291320925213408, 1.0875719215668045, 0.9650947816171573, 0.9117180742670491, 1.005666896790334, 1.0021405454440135, 0.9411422930877907, 0.9196215386006089, 0.993155898897068, 1.008090895388381, 1.0204931458479785, 0.9702777700665935, 0.9855823501011933, 1.1459242484950645, 1.116720677456746, 1.1085546947399612, 1.099318118747002,1.20] #M062x to be updated
   MAGNAVE=[1.0, 1.2777868476919305, 1.1070427070723463, 1.1311173540061936, 1.0764121980959842, 1.151123069185254, 1.099181786330849, 1.1398157635768424, 1.1331148767870864, 1.098386851481681, 1.054531675013483, 1.1973823414210398, 1.004551493948572, 1.0581077759825357, 1.0395620076931287, 1.1800767886351171, 1.0359623468310752, 1.0087376307954916, 1.2072497549617507, 1.1648490664313624, 1.0202068977630299, 1.0113727965218904, 1.0061301149812445, 1.177520706120571, 1.024998585494763, 1.0405024134074636, 1.282349995531235, 1.2524684371118955, 1.0676953539315441, 1.056071099690981, 1.172132291387645, 1.1668799549267759, 1.2399345004898705, 1.070031538713933, 1.1000649842005894, 1.1644370815913037, 1.1803017367960955, 1.1668975621157645, 1.0157060519106205, 1.0889579917399226, 0.9966654801904894, 1.1729751731351428, 1.0253774912018085, 1.0590860365172026, 1.0953646216355721, 1.1699728475073916, 1.0507622618098413, 1.0650953183933172, 1.1855913694813547, 1.0112315129277443, 1.0293549811071216, 1.1291220405126978, 1.1400034220225408, 1.0078135361317706, 1.0156423940172523, 1.054658323443067, 1.1573632110026055, 1.0092836666702303, 1.1663363181705686, 0.9925708297611614, 1.0196787038170523, 1.0112443450480708, 1.2014006851663857, 1.0336445183787941, 1.0725089363703726, 1.0390411934458335, 0.9712213084943183, 1.0271528543892898, 1.030996660799963, 0.9748910205158364, 0.9619677783081638, 1.2318064630130303, 1.1814173198401587, 0.9891411057574929, 0.9203776253873474, 1.0199726280717736, 1.0364068208113562, 0.9609333551221461, 0.934647601551232, 1.0754130634203973, 1.0993329752049272, 1.138960176504743, 1.0795850723816003, 1.0916008140218396, 1.259238896072771, 1.1137649350787355, 1.0931115604731756, 1.0692274921726344,1.20] #M062x to be updated

   if fitting == "dft":
#      if bas == bas0 : always use this 
      print("Basis set vector match")
      index = DFTLIST.index(dft)
      index0= DFTLIST.index(dft0)
      print("Target : ", DFTLIST[index]," and reference : ",DFTLIST[index0]) 
      if(aimming=='tot'):
         dv = MAGNTOT[index]/MAGNTOT[index0]
      elif(aimming=='ave'):
         dv = MAGNAVE[index]/MAGNAVE[index0]

#      elif dft == dft0 :
#         print("DFT func. vector match")
#         Nbasis=getNbasis(bas)
#         print(Nbasis)
      else :
         print("None match ")
   elif fitting == "ploy2-gaugms":
      if(aimming=='tot'):
         dv = 0.55 
   else :
      if dft == dft0: 
         print("DFT func. vector match")

 
   print(dft,bas,dft0,bas0,": dv = ",dv)
   return dv


def equals(st1:str,st2:str):
   if st1.upper()==st2.upper():
      return True
   st11=set(st1.upper())
   st22=set(st2.upper())
   list1=list(st11^st22)
   if len(list1)==1 and list1[0]=='-':
      return True
   return False

# the correction coefficient comes from the time of one same basis
def magnification2(cs,cs0,aimming='tot'):
   idx=-1
   if aimming=='tot':
      idx=0
   elif aimming=='ave':
      idx=1
   else:
      raise Exception("aimming should be 'tot' or 'ave' ")  

   dft=cs.split("_")[0]
   bas=cs.split("_")[1]
   dft0=cs0.split("_")[0]
   bas0=cs0.split("_")[1]

   bas=bas.replace("s","*").replace("p","+").replace("+V","pV")
   bas0=bas0.replace("s","*").replace("p","+").replace("+V","pV")
   dv=100.0

   if bas==bas0:
      print("Basis set vector match")
      jPath=PREFIX+"coefDict_"+dft0+".json"
      with open(jPath,'r') as f:
         coef_dict=json.load(f)
      dfts=coef_dict.keys()
      for key in dfts:
         if equals(dft,key):
            dft=key
            for bkey in coef_dict[key].keys():
               if equals(bkey,bas):
                  bas=bkey 
      dv=coef_dict[dft][bas][idx]
   elif dft==dft0:
      print("DFT func. vector match")
   else :
      print("None match ")
   
   print(dft,bas,dft0,bas0,": dv = ",dv)
   return dv


# print(magnification2('bhandhlyp_6-31+G*','PBE1PBE_6-31+G*','tot'))
