from gensim.models import Word2Vec
import gensim
import sys

sys.path.append("D:/Fcst/fcst_sys")

from LstmTool2 import LstmTool2
from rdkit import Chem
import os

#wvmodel=gensim.models.KeyedVectors.load_word2vec_format("D:/Fcst/fcst_sys/word2Vec.bin",binary=True,encoding='utf-8')
# res=wvmodel.most_similar('C',topn=5)
# for item in res:
#     print(item[0]," ",item[1])

# sdfs=[]
# path="D:/Fcst/fcst_sys/tools/test_suits_sdf/DrugBank/sdf_folder"
# sdfs.extend(os.listdir(path))
# for i in range(len(sdfs)):
#     tmp=sdfs[i]
#     sdfs[i]=path+"/"+tmp

# smiles=[]

# for item in sdfs:
#     sup=Chem.SDMolSupplier(item)
#     for mol in sup:
#         if mol==None:
#             continue
#         smi=Chem.MolToSmiles(mol)
#         smiles.append(smi)


# clist=LstmTool2.seg(smiles)
# wvmodel=gensim.models.Word2Vec(sentences=clist,sg=1,window=3)
# print(wvmodel.similarity('C','O'))
#wvmodel.save("D:/Fcst/fcst_sys/word2Vec_2")
#wvmodel.save("D:/Fcst/fcst_sys/word2Vec_2.bin")
#wvmodel.wv.save_word2vec_format("D:/Fcst/fcst_sys/word2Vec_2",binary=False)

wvmodel=gensim.models.KeyedVectors.load_word2vec_format("D:/Fcst/fcst_sys/word2Vec_2.bin",binary=True,encoding='utf-8')
res=wvmodel.most_similar('N',topn=10)
for item in res:
    print(item[0]," ",item[1])