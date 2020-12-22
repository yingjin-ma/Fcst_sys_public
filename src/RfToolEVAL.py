import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import re
import os
from rdkit import Chem
from rdkit.Chem import AllChem
# If sklearn < 0.21+
#from sklearn.externals import joblib
# If sklearn > 0.21+
import joblib
import numpy as np
from FNN2 import FNN
import torch.optim as optim
import torch.nn as nn
import xlsxwriter
from ModelTool import ModelTool

import basis_set_exchange as bse
from Magnification import getNbasis
import TrainedMods

torch.manual_seed(2)

#RF class
class RfTool(ModelTool):

    clf=None #classify

    def __init__(self,chemspace,config,suits1,sdf_dir,target,BAK):
        '''
        construction
        chemspace
        config :  need Configs.Config to initialize
        '''
        ModelTool.__init__(self,chemspace,config,sdf_dir,target)
        self.suits1=suits1
        
        # if os.path.exists(BAK+"/RFC.m"):
        #     self.clf = joblib.load(BAK+"/RFC.m")
        if TrainedMods.mod_exists("RFC"):
            self.clf = TrainedMods.getModel("RFC")

    #     alkene - 1 
    #     branch - 2 
    #       ring - 3 
    #    polyene - 4
    # polyphenyl - 5

    @staticmethod
    def atom_count(smiles):
        '''
        return number of atoms (except H atom) in smiles
        '''
        atom_num=0
        atomset={'B','C','N','O','F','P','S','I','c'}
        for elem in atomset:
            atom_num=atom_num+smiles.count(elem)
        return atom_num   

    @staticmethod
    def smiles_analyzer(smiles):
        '''
        return number of atoms
        nunmber of branches, number of atoms in branches, 
        number of rings    , numner of atoms in rings
        number of C=C bonds
        (Except the H atom)
        '''
        atom_num=RfTool.atom_count(smiles)
        branch_num=smiles.count('(')
        branch_atom_num=0
        cycle_atom_num=0
        p=re.compile(r'[(](.*?)[)]', re.S) # extract the content in bra ket
        s=re.findall(p,smiles)
        s=''.join(s)
        s=s.strip()
        s=re.sub(r'\W','',s)
        s=re.sub(r'\d','',s)
        branch_atom_num=RfTool.atom_count(s)
        cycle_num=0
        for s in smiles:
            if s.isdigit():
                cycle_num=cycle_num+1
        cycle_num=int(cycle_num/2)
        # In case no ring in smiles, or the ring number only exist in one ring 
        if smiles.count('1')<=2: 
            for i in range(1,cycle_num+1):
                start=smiles.find(str(i),0)
                end=smiles.find(str(i),start+1)
                cycle=smiles[start:end]
                cycle=''.join(re.split(r'[^A-Za-z]',cycle))
                # add the atoms before the start position
                cycle_atom_num=cycle_atom_num+RfTool.atom_count(cycle)+1 
                if cycle_atom_num>atom_num:
                    # to avoid the redundent counters if ring-in-ring case
                    cycle_atom_num=cycle_atom_num-RfTool.atom_count(cycle)-1
        else:
            # in case the '1' for rings is counted more than 2 times
            begin=smiles.find('1',0)
            for i in range(1,cycle_num+1):
                start=begin
                end=smiles.find('1',start+1)
                begin=smiles.find('1',end+1)
                cycle=smiles[start:end]
                cycle=''.join(re.split(r'[^A-Za-z]',cycle))
                cycle_atom_num=cycle_atom_num+RfTool.atom_count(cycle)+1
                if cycle_atom_num>atom_num:
                    cycle_atom_num=cycle_atom_num-RfTool.atom_count(cycle)-1
        double_bonds=smiles.count('=')
        double_bonds-=smiles.count('=O')
        return {'atom_num':atom_num,'branch_num':branch_num,'branch_atom_num':branch_atom_num,'cycle_num':cycle_num,'cycle_atom_num':cycle_atom_num,
        'double_bonds':double_bonds}

    @staticmethod
    def smiles_to_ft(slist):
        '''
        extract the smiles in suits 
        '''
        data=[]
        for s in slist:
            dic=RfTool.smiles_analyzer(s)
            atom_num=np.int32(dic['atom_num'])
            branch_num=np.int32(dic['branch_num'])
            branch_atom_num=np.int32(dic['branch_atom_num'])
            cycle_num=np.int32(dic['cycle_num'])
            cycle_atom_num=np.int32(dic['cycle_atom_num'])
            double_bonds=np.int32(dic['double_bonds'])
            data.append([atom_num, branch_num, branch_atom_num, cycle_num, cycle_atom_num,double_bonds])
        return data    


    @staticmethod
    def gen_data_for_rf():
        '''
        generate the data for RF models 
        '''        
        data=[]
        s='CCCC'
        for i in range(200):
            s+='C'
            if i%25==0:
                s+='(=O)C'
            if i%11==0:
                s+='(C)C'
            data.append([s,1])
        s='CCCC'
        for i in range(200):
            if i<40:
                s+='(CCC)CC'
            else:
                s+='(CCCC)C'
            
            if i%7==0:
                s+='(=O)C'
            data.append([s,2])
        s='C1CCCCC1'    
        for i in range(200):
            if i<20:
                s+='CC1CCCC1C'
            elif i<40:
                s+='CO1CCCCC1C'
            elif i<50:
                s+='CC1CC(C)CCC1C'
            if i%7==0:
                s+='CC1CC(=O)CC1C'
            if i%19==0:
                s+='c1ccccc1'
            data.append([s,3])
        s='C=CC=C'
        for i in range(200):
            s+='C=C'
            if i%10==0:
                s+='C(=O)C=C'
            if i%19==0:
                s+='C(C)C=C'
            data.append([s,4])
        data=np.array(data)
        np.random.shuffle(data)
        with open('tra_clf_data.txt',mode='w',encoding='utf-8') as f:
            for i in range(800):
                line=str(data[i][0])+' '+str(data[i][1])+'\n'
                f.write(line)



    @staticmethod
    def read_data_for_rf(path): 
        '''
        training data for classify
        '''    
        # Each term : 
        #       0 - class label   
        #       1 - atoms   
        #       2 - number of branches  
        #       3 - number of atoms in branches (except H)  
        #       4 - number of rings  
        #       5 - number of atoms in rings (except H)  
        #       6 - number of C=C bonds
        data=[]
        with open(path,'r',encoding='utf-8-sig') as sfile:
            lines=sfile.readlines()
            for line in lines:
                L=line.strip(os.linesep).split(' ')
                dic=RfTool.smiles_analyzer(L[0])
                atom_num=np.int32(dic['atom_num'])
                branch_num=np.int32(dic['branch_num'])
                branch_atom_num=np.int32(dic['branch_atom_num'])
                cycle_num=np.int32(dic['cycle_num'])
                cycle_atom_num=np.int32(dic['cycle_atom_num'])
                double_bonds=np.int32(dic['double_bonds'])
                class_tag=np.int32(L[1])
                data.append([class_tag, atom_num, branch_num, branch_atom_num, cycle_num, cycle_atom_num,double_bonds])
        return data

    
    @staticmethod
    def train_rf(data):
        '''
        RF calssify, data as list
        '''
        clf=RandomForestClassifier(oob_score=True,random_state=10)
        x=[]
        y=[]
        for a in data:
            x.append([a[1],a[2],a[3],a[4],a[5],a[6]])
            y.append(a[0])    
        clf.fit(x,y)
        joblib.dump(clf, "../model/RFC.m")
        return clf

    @staticmethod
    def readData(paths,sdf_dir,tra_size,target,basis="",readName=True):
        '''
        read in the training or testing suits
        paths    : paths for data
        sdf_dir  : sdf folder
        target   : target 
        readName : filename of sdf 
        tra_size : training size 
        '''

        print("sdf_dir in readData",sdf_dir) 

        basisnums =[]
        times=[]
        names=[]
        slist=[]
        count=0
        for path in paths:
            with open(path,'r') as f:
                lines=f.readlines()
                for line in lines:
                    #print(line)
                    count+=1
                    if count>tra_size:
                        break
                    temp=line.strip(os.linesep).split()
                    #print(temp)

                    sdf=sdf_dir + "/" + temp[4].split('_')[0]+".sdf"

                    if float(temp[1]) == 0 :
                      basisnum=float(temp[0])
                    else:  
                      basisnum=float(temp[1])

                    #else  :
                    basisnums.append(basisnum)
                        
                    time=float(temp[target])    
                    times.append(time)
                    if readName==True:
                        name=temp[4].split("_")[0]
                        names.append(name)
                        s=sdf_dir+'/'+name+'.sdf'
                        suppl=Chem.SDMolSupplier(s)
                        for mol in suppl:
                            smiles=Chem.MolToSmiles(mol)
                            slist.append(smiles)
        return basisnums,times,slist,names

    @staticmethod
    def sdf_to_smiles(sdfs):#sdfs=[sdf1,sdf2,...]
        '''
        sdf  ==>  smiles
        sdfs: sdf list with the format of sdfs=[sdf1,sdf2,...]
        '''
        slist=[]
        for sdf in sdfs:
            suppl=Chem.SDMolSupplier(sdf)
            for mol in suppl:
                smiles=Chem.MolToSmiles(mol)
                slist.append(smiles)
        return slist


    def classify(self,x):
        '''
        RF classify, return the ratio for every types of molecules/fragments
        x: to be predicted,  in the format of [[18,0,0,1,18,0]]
        '''      
        if self.clf==None:
            print("classifier has not been loaded")
            return 
        x=np.array(x)
        return self.clf.predict_proba(x)  


    def eval_single(self,path,moltype,modelloc=None):
        '''
        Test the specific FN model
        moltype: 1 - alk, 2 - branch, 3 - ring, 4 - pe
        '''
        assert moltype in (1,2,3,4)
        tra_size=self.config.tra_size
        basisnums,times,slist,names=RfTool.readData(path,self.sdf_dir,tra_size,target=self.target)
        struct_fts=RfTool.smiles_to_ft(slist)#struc_fits: [[],[],...]
        feats=[]
        for i in range(len(basisnums)):
            feat=[]
            feat.append(basisnums[i])
            feat.extend(struct_fts[i])
            feats.append(feat)
        feats_t=torch.tensor(feats)
        times_t=torch.tensor(times)
        modelName=''
        if modelloc==None:
            modelName= 'rfmodel/'+self.chemspace+'_'+moltype+'.pkl'
        else:
            modelName=modelloc
        model=torch.load(modelName)
        model.eval()
        model=model.to(self.device)
        eval_set=torch.torch.utils.data.TensorDataset(feats_t,times_t)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=self.config.batch_size,shuffle=False)
        with torch.no_grad():
            mre=0
            mae=0
            for ft,time in eval_iter:
                ft=ft.to(self.device)
                tmp=model(ft)
                reslist=tmp.to('cpu').numpy().tolist()
                ftlist=ft.to('cpu').numpy().tolist()
                timelist=time.numpy().tolist()
                print(reslist)
                print(ftlist)
                print(timelist)
                j=0
                for i in range(len(reslist)):
                    j+=1
                    ae=abs(reslist[i]-timelist[i])
                    re=ae/timelist[i]
                    #print("基组: %d, 预测值: %.5f, 真实值: %.5f"%(ft[i][0],reslist[i],timelist[i]))
                    mae+=ae
                    mre+=re
            mre/=j
            mae/=j
            print("mre: %.5f, mae: %.5f"%(mre,mae))
            return [mre,mae]        

    def train(self,path,moltype,modelloc):
        '''
        训练FNN
        path: 训练集路径
        moltype: 模型类别，规定直链烷烃分子类标签为1,支链分子为2,环状分子为3，直链烯烃为4
        '''
        assert moltype in (1,2,3,4)
        tra_size=self.config.tra_size
        basisnums,times,slist,names=RfTool.readData(path,self.sdf_dir,tra_size,target=self.target)
        struct_fts=RfTool.smiles_to_ft(slist)#struc_fits: [[],[],...]
        feats=[]
        for i in range(len(basisnums)):
            feat=[]
            feat.append(basisnums[i])
            feat.extend(struct_fts[i])
            feats.append(feat)
        t_feats=torch.tensor(feats)
        t_times=torch.tensor(times)

        model=FNN(hidden_dim=self.config.hidden_dim,output_dim=1,num_layers=self.config.num_layers)

        loss_function=nn.L1Loss()
        optimizer=optim.Adam(model.parameters(),lr=self.config.lr)
        
        model.train()
        model.to(self.device)
        
        train_loss=0
        save_step=100
        # if self.config.tra_num_epochs>1000:
        #     save_step=500

        minMre=100.0
        bestEpoch=0
        for epoch in range(self.config.tra_num_epochs):
            j=0
            mre=0.0
            model.zero_grad()
            t_feats=t_feats.to(self.device)
            t_times=t_times.to(self.device)
            result=model(t_feats)   
            loss=loss_function(result,t_times)
            train_loss=loss
            loss.backward()
            optimizer.step()
            reslist=result.to('cpu').detach().numpy()
            reslist=reslist.tolist()
            timelist=t_times.to('cpu').detach().numpy()
            timelist=timelist.tolist()
            for i in range(len(reslist)):
                j+=1
                ae=abs(reslist[i]-timelist[i])
                re=ae/timelist[i]
                mre+=re
            mre/=j
            print("epoch: %d, loss: %.5f, mre: %.5f"%(epoch,train_loss,mre))
            modelloc_tmp=modelloc.split('.')[0]+'_tmp.pkl'
            if epoch%save_step==0:
                torch.save(model,modelloc_tmp)
                eval_res=self.eval_single(path,moltype,modelloc_tmp)
                if eval_res[0]<minMre:
                    torch.save(model,modelloc)
                    minMre=eval_res[0]
                    bestEpoch=epoch       
        # if tra_size<4000:
        #     moltype=str(moltype)+'_'+str(tra_size)
        # if modelloc==None:    
        #     torch.save(model,'rfmodel/'+self.chemspace+'_'+moltype+'.pkl')
        # else:
        #     torch.save(model,modelloc)
        print("trainning done! best epoch is "+str(bestEpoch))


    

        
    #测试分类器
    def eval_clf(self,path):
        tra_size=self.config.tra_size
        basisnums,basisnums2,times,slist,names=RfTool.readData(paths=path,sdf_dir=self.sdf_dir,tra_size=tra_size,target=self.target)
        struct_fts=RfTool.smiles_to_ft(slist)#struc_fits: [[],[],...]
        struct_fts=np.array(struct_fts)
        probs=self.clf.predict_proba(struct_fts)#probs: [[] [] []] np array
        for i in range(len(basisnums)):
            print("Name ",names[i],probs[i])

    # testing
    def eval(self,modelname=None,path=None,chemspace="B3LYP_6-31g",mol="sample.sdf",write=False,BAK=None):
#    def eval(self,path=None,modeldir='rfmodel',chemspace="B3LYP_6-31g",write=False):
        '''
        FNN model test
        path: path for testing suits 
        write: if write into xlsx tables
        '''
        modeldir=modelname
        #print("modeldir",modeldir)
        #print("chemspace",chemspace)
        dft  =chemspace.split("_")[0]
        basis=chemspace.split("_")[1]

        molecule = path+"/"+mol
        nbasis   = getNbasis(bas=basis,sdf=molecule)
        #print(" nbasis ", nbasis, " path ",path)

        assert self.clf!=None

        #if path==None:
        #    path='./data/tes_'+self.chemspace+'.txt'
            
        tra_size=self.config.tra_size
        #basisnums,times,slist,names=RfTool.readData(paths=path,sdf_dir=self.sdf_dir,tra_size=tra_size,target=self.target,basis=basis)

        basisnums=[]
        times=[]
        slist=[]
        names=[]
        basisnums.append(nbasis*1.0)
        times.append(1.0)
        names.append(mol)
        suppl=Chem.SDMolSupplier(molecule)
        for imol in suppl:
           smiles=Chem.MolToSmiles(imol)
           slist.append(smiles)

        # debug or check 
        #print("basisnums",basisnums)

        struct_fts=RfTool.smiles_to_ft(slist)#struc_fits: [[],[],...]
        struct_fts_np=np.array(struct_fts)
        probs=self.clf.predict_proba(struct_fts_np)#probs: [[] [] []] np数组
        
        feats=[]
        for i in range(len(basisnums)):
            feat=[]
            feat.append(basisnums[i])
            feat.extend(struct_fts[i])
            feats.append(feat)
        feats_t=torch.tensor(feats)
        times_t=torch.tensor(times) 
        #slist_t=torch.tensor(slist)
        #names_t=torch.tensor(names)
        probs_t=torch.from_numpy(probs)

        preds=[]


        models=[]
        for i in range(0,4):
            #model=torch.load(modeldir+'/'+self.chemspace+'_'+str(i+1)+'.pkl',map_location=self.device)
            model=TrainedMods.getModel(modeldir+'/'+self.chemspace+'_'+str(i+1)+'.pkl')
            model.eval()
            #model.to(self.device)
            models.append(model)

        eval_set=torch.torch.utils.data.TensorDataset(feats_t,times_t,probs_t)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=self.config.batch_size,shuffle=False)

        mae=0.0
        count=0
        with torch.no_grad():
            errs=[]

            for ft,time,proba in eval_iter:
                res=[]
                ft=ft.to(self.device)
                #time=time.to(self.device)
                proba=proba.to(self.device) # size=n x 4
                #print("proba: ",proba)
                for i in range(0,4):
                    temp=models[i](ft) #temp=tensor([v1,v2,....])
                    res.append(temp) #res=[temp1,temp2,temp3,temp4]
                res=torch.stack(res)#res=tensor(temp1,temp2,temp3,temp4) size=4 x n
                res=torch.transpose(res,0,1) # n x 4

                pred=torch.mul(proba,res)
                pred=pred[:,0]+pred[:,1]+pred[:,2]+pred[:,3]
                
                pred=pred.to('cpu')
                predlist=pred.numpy().tolist()
                timelist=time.numpy().tolist()
#                for i in range(len(predlist)):
#                    predlist[i]=predlist[i] 
#                    err1=(float(timelist[i])-float(predlist[i]))/float(timelist[i])
#                    #print('i: ',i, ' sdf/mol: ', (names[count]) ,' basis num: ',basisnums[i],' real time : ',timelist[i],' predicted time: ',predlist[i], 'err', err1)
#                    #print("名称: %s, 预测值: %.5f, 真实值: %.5f"%(names[count],predlist[i],timelist[i]))
#                    count+=1
#                    preds.append(predlist[i])
#                    ae=abs(predlist[i]-timelist[i])
#                    err=ae/timelist[i]
#                    errs.append(err)
#                    mae+=ae

            #errs=np.array(errs)
            #mre=errs.mean()
            #var=errs.var()
            #mae/=len(basisnums)

            #print("MRE: %.5f, mae: %5.f, VAR: %.5f"%(mre,mae,var))

        return predlist


    def predict(self,pdata,batch_size):#pdata:[[sdf1,sdf2,...],[nbasis1,nbasis2,...]]
        '''
        返回模型推断结果
        pdata: 待预测数据,形式为[[sdf1,sdf2,...],[nbasis1,nbasis2,...]]
        batch_size: batch的大小
        '''
        assert self.clf!=None
        slist=RfTool.sdf_to_smiles(pdata[0])
        bnums=pdata[1]
        preds=[]

        struct_fts=RfTool.smiles_to_ft(slist)
        struct_fts=np.array(struct_fts)
        probs=self.clf.predict_proba(struct_fts)#probs: [[] [] []] np数组

        bnums_t=torch.tensor(bnums)
        probs_t=torch.from_numpy(probs)

        models=[]
        for i in range(0,5):
            model=torch.load('rfmodel/'+self.chemspace+'_'+(i+1)+'.pkl')
            model.eval()
            model.to(self.device)
            models.append(model)

        eval_set=torch.torch.utils.data.TensorDataset(bnums_t,probs_t)
        eval_iter=torch.utils.data.DataLoader(eval_set,batch_size=batch_size,shuffle=False)

        with torch.no_grad():
            for bnum,proba in eval_iter:
                res=[]

                bnum=bnum.to(self.device)
                #time=time.to(self.device)
                proba=proba.to(self.device)

                for i in range(0,5):
                    temp=models[i](bnum) #temp=tensor([v1,v2,....])
                    res.append(temp) #res=[temp1,temp2,...temp5]

                res=torch.stack(res)#res=tensor(temp1,temp2,...)
                res=torch.transpose(res,0,1)

                pred=torch.mul(proba,res)
                pred=pred[:,0]+pred[:,1]

                pred=pred.to('cpu').squeeze(dim=1)
                predlist=pred.numpy().tolist()
                preds.extend(predlist)

        return preds
                



        

        



                



                

        


    



    
