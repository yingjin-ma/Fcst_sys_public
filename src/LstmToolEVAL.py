from rdkit import Chem
import json
import os
import torch
import torch.nn as nn
import gensim
import BiLSTM
import time as systime
import torch.optim as optim
import numpy as np
import xlsxwriter
from ModelTool import ModelTool

torch.manual_seed(2)

import basis_set_exchange as bse
from Magnification import getNbasis, getNbasis_noRDkit

# LSTM模型的工具类，用于模型的训练、测试、推断
short_sdf_index = []

class LstmTool(ModelTool):

    def __init__ (self,chemspace,config,suits1,sdf_dir,target):
        '''
        construction
        chemspace
        config :  need Configs.Config to initialize
        '''
        ModelTool.__init__(self,chemspace,config,sdf_dir,target)
        self.suits1=suits1

    @staticmethod
    def readData(paths, sdf_dir,tra_size,target,basis=""): 
        '''
        Get the data, Nbasis - CPU times - smiles - sdf names, from suits
        '''
        print("sdf_dir in readData",sdf_dir)
        print("paths in readData",paths)

        basisnums = []
        times = []
        slist = []  # smiles list
        names = []
        count = 0
        for path in paths:
            for line in open(path,'r'):
                count+=1
                if count>tra_size:
                    break
                temp=line.strip(os.linesep).split()
                time=float(temp[target])#时间

                sdf=sdf_dir + "/" + temp[4].split('_')[0]+".sdf"
                for i in range(len(temp)):
                    if temp[i] == 'contracted':
                        basisnum_s = float(temp[i+2])
                        basisnum_p = float(temp[i+3])
                        basisnum_d = float(temp[i+4])
                        basisnum_f = float(temp[i+5])
                        basisnum_g = float(temp[i+6])
                        basisnum_h = float(temp[i+7].strip(']'))
                        break

                basisnums = [basisnum_s, basisnum_p, basisnum_d, basisnum_f, basisnum_g, basisnum_h]#各个轨道总数目
                name=temp[4]#.split('_')[1]
                names.append(name)
                times.append(time)
                s=sdf_dir+'/'+name.split('_')[0]+'.sdf'
                suppl=Chem.SDMolSupplier(s)
                for mol in suppl:
                    smiles=Chem.MolToSmiles(mol)
                    slist.append(smiles)
        return basisnums,times,slist,names


    @staticmethod
    def sdf_to_smiles(sdfs): 
        '''
        sdf ==> smiles
        '''
        slist=[]
        for sdf in sdfs:
            suppl=Chem.SDMolSupplier(sdf)
            for mol in suppl:
                smiles=Chem.MolToSmiles(mol)
                slist.append(smiles)
        return slist


    @staticmethod
    def seg(slist):
        '''
        smiles ==> words
        '''
        clist=[]
        for smiles in slist:
            smile=list(smiles)
            clist.append(smile)
        return clist

    @staticmethod
    def genDict():
        '''
        generate words and dict
        '''
        #words in smiles
        vocab=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h',
        'i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','+','-',
        '[',']','(',')','.','=','#','$',':','/','\\','@']
        word_to_idx = {word: i+1 for i, word in enumerate(vocab)}  #  word - index dict
        word_to_idx['?'] = 0                                       #       unknown word
        idx_to_word={i+1: word for i, word in enumerate(vocab)}    # index -  word dict
        idx_to_word[0] = '?'
        with open("wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(word_to_idx,f)
        with open("indexToWord.json","w",encoding="utf-8") as f:
            json.dump(idx_to_word,f)
        return word_to_idx,idx_to_word

    @staticmethod
    def wToIdx(clist,word_to_idx):
        '''
        smiles ==> index
        '''
        features=[]
        for smiles in clist:
            feature=[]
            for s in smiles:
                if s in word_to_idx:
                    feature.append(int(word_to_idx[s]))
                else:
                    feature.append(0)
            features.append(feature)
        return features


    @staticmethod
    def pad(features,maxlen=200,PAD=0):
        '''
        fix length of every molecules to 100
          > 100 : cut off
          < 100 : add   0  
        '''
        padded_features=[]
        for feature in features:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
            padded_features.append(padded_feature)
        return padded_features

    def evalsuit(self, modelname=None, path=None, chemspace="B3LYP_6-31g", BAK=None, write=False):
        '''
        Predicting
        path: path of testing suits; default as 'data/tes_'+self.chemspace+'.txt'
        write: check whether into xlsx format
        '''

        dft = chemspace.split("_")[0]
        basis = chemspace.split("_")[1]

        # tra_size=self.config.tra_size
        print("self.suits1  : ", self.suits1)
        print("self.sdf_dir : ", self.sdf_dir)

        mollist = []
        baslist = []
        basisnums = []
        times = []
        slist = []
        names = []
        basisnumlist = []
        suppl = []
        for isuit in self.suits1:
            print("isuit : ", isuit)
            imol = self.sdf_dir + "/" + isuit
            # ibas = getNbasis(bas=basis, sdf=imol)
            # obasis, nbasis = getNbasis(bas=basis, sdf=imol)
            obasis, nbasis = getNbasis_noRDkit(bas=basis,sdf=imol)
            print("imol : ", imol, " ibas : ", nbasis)
            mollist.append(imol)
            baslist.append(nbasis)
            basisnumlist.append(obasis)

            basisnums.append(nbasis * 1.0)
            times.append(1.0)
            names.append(isuit)
            suppl.append(Chem.SDMolSupplier(imol))

            # print("Done the suppl")

        pdata = [mollist,basisnumlist, baslist]

        i = 0
        for imol in suppl:
            if imol is None:
                names.pop(i)
                basisnums.pop(i)
                times.pop(i)
                basisnumlist.pop(i)
                i += 1
                continue
            smiles = Chem.MolToSmiles(imol)
            i += 1
            slist.append(smiles)
        model = torch.load(modelname)
        model = model.to(self.device)
        model.eval()


        clist = LstmTool.seg(slist)
        with open(BAK + '/wordToIndex.json', 'r', encoding='utf8') as f:
            word_to_idx = json.load(f)
        features = LstmTool.wToIdx(clist, word_to_idx)
        padded_features = LstmTool.pad(features)
        eval_features = torch.tensor(padded_features)
        eval_basis_s = torch.tensor(basisnums)
        eval_time = torch.tensor(times)
        eval_basis = torch.tensor(basisnumlist)


        eval_set = torch.utils.data.TensorDataset(eval_features, eval_basis, eval_basis_s, eval_time)
        eval_iter = torch.utils.data.DataLoader(eval_set, batch_size=self.config.batch_size, shuffle=False)

        ij = 0
        preds = []
        with torch.no_grad():

            err_mean = 0.0
            errs = []
            j = 0

            ae = 0.0
            for feature,basisnum,basisnums,time in eval_iter:
                # j=0
                feature = feature.to(self.device)
                basisnum = basisnum.to(self.device)
                basisnums = basisnums.to(self.device)

                result = model(feature, basisnum, basisnums)
                result = result.to('cpu')
                resultlist = result.numpy().tolist()
                preds.extend(resultlist)
                basislist = basisnum.to('cpu').numpy().tolist()
                timelist = time.numpy()
                timelist = timelist.tolist()

                ij = ij + 1
                print("ij : ", ij)


        i = 0
        print("len(names)", len(names), "len(preds)", len(preds))
        for isuit in self.suits1:
            print(i + 1, " ", names[i], " ", preds[i])
            i = i + 1
            if i >= len(names):
                break

        return resultlist

    def eval(self,modelname=None,path=None,chemspace="B3LYP_6-31g",mol="sample.sdf",write=False,BAK=None):
        '''
        testing or evaluating 
        path: path of testing suits; default as'data/tes_'+self.chemspace+'.txt'
        write: check whether into xlsx format
        '''

        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]

#        if path==None:
#            path=['data/tes_'+self.chemspace+'.txt']
#        tra_size=self.config.tra_size

        molecule = path+"/"+mol
        obasis, nbasis = getNbasis(bas=basis, sdf=molecule)
        print(" nbasis ", nbasis)
        pdata = [[molecule], [obasis], [nbasis]]

#        if modelname==None:
#            modelname=self.modelloc
#        
#        if not os.path.exists(modelname):
#            print(modelname+" does not exist!")
#            return

        model=torch.load(modelname)
        model=model.to(self.device)
        model.eval()
        
        #basisnums,times,slist,names=LstmTool.readData(path,self.sdf_dir,tra_size,self.target,basis=basis)

        basisnum = []
        basisnums=[]
        times=[]
        slist=[]
        names=[]
        basisnum.append(obasis)
        basisnums.append(nbasis)
        times.append(1.0)
        names.append(mol)
        suppl=Chem.SDMolSupplier(molecule)
        for imol in suppl:
           smiles=Chem.MolToSmiles(imol)
           slist.append(smiles)

        clist=LstmTool.seg(slist)
        with open('./tmp/wordToIndex.json','r',encoding='utf8') as f:
            word_to_idx=json.load(f)
        features=LstmTool.wToIdx(clist,word_to_idx)
        padded_features=LstmTool.pad(features)
        eval_features=torch.tensor(padded_features)
        eval_basis=torch.tensor(basisnum)
        eval_basis_s = torch.tensor(basisnums)
        eval_time=torch.tensor(times)
        #eval_names=torch.tensor(names)
        #nameDic={idx:name for idx,name in enumerate(names)}
        #eval_names=torch.tensor(nameDic.keys())

        eval_set=torch.utils.data.TensorDataset(eval_features,eval_basis,eval_basis_s,eval_time)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=self.config.batch_size,shuffle=False)

        preds=[]
        with torch.no_grad():

            #with open(tmp1,'r') as ftmp:
               #ftmplines=ftmp.readlines()

            err_mean=0.0
            errs=[]
            j=0

            ae=0.0
            for feature,basisnum,basisnums,time in eval_iter:
                #j=0
                feature = feature.to(self.device)
                basisnum = basisnum.to(self.device)
                basisnums = basisnums.to(self.device)

                result=model(feature,basisnum,basisnums)
                result=result.to('cpu')
                resultlist=result.numpy().tolist()
                preds.extend(resultlist)
                basislist = basisnum.to('cpu').numpy().tolist()
                basislist_s = basisnums.to('cpu').numpy().tolist()
                timelist = time.numpy()
                timelist = timelist.tolist()
                #print("len(resultlist)",len(resultlist),"len(ftmplines)",len(ftmplines),"len(basisnums2)",len(basisnums2)) 
#                for i in range(len(resultlist)):
#                    #print("基组: %d, 预测值: %.5f, 预测值(corrected): %.5f, 真实值: %.5f"%(basislist[i],resultlist[i],resultlist[i]*dv_magn,timelist[i]))
#                    resultlist[i]=resultlist[i]
#                    j+=1
#                    err1=(float(resultlist[i])-float(timelist[i]))/float(timelist[i])
#                    #print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',basislist[i],' real time : ',timelist[i],' predicted time: ',resultlist[i], 'err', err1)
#                    single_err=abs((resultlist[i]-timelist[i])/timelist[i])
#                    ae=ae+abs(resultlist[i]-timelist[i]) 
#                    errs.append(single_err)
#                    err_mean+=single_err
            #err_mean=err_mean/j
            #mae=ae/j
            #errs=np.array(errs)
            #variance=errs.var() 

            #print("MRE：%.4f, MAE: %.4F, 方差: %.4f"%(err_mean,mae,variance))

        return resultlist
    
