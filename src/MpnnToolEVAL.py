import argparse
import torch as th
import torch.nn as nn
import numpy as np
from MPNN import MPNNModel
from torch.utils.data import DataLoader
from GDataSetEVAL import TADataset, batcher
import os
import xlsxwriter
from ModelTool import ModelTool

th.manual_seed(2)

import basis_set_exchange as bse
from Magnification import getNbasis

# MPNN模型的工具类
class MpnnTool(ModelTool):
    
    def __init__ (self,chemspace,config,suits1,sdf_dir,target):
        '''
        construction
        chemspace
        config :  need Configs.Config to initialize
        '''
        ModelTool.__init__(self,chemspace,config,sdf_dir,target)
        self.suits1=suits1

    def eval(self,modelname,chemspace="B3LYP_6-31g",path='./',mol="sample.sdf",write=False):
        '''
        testing model
        path: data upper folder, default folder, data with data
        write:  can be xlsx file
        '''
        dft      = chemspace.split("_")[0]
        basis    = chemspace.split("_")[1]
        #print("dft ", dft ,"  basis", basis)        
        tra_size = self.config.tra_size
        molecule = path+"/"+mol
        nbasis   = getNbasis(bas=basis,sdf=molecule)
        print(" nbasis ", nbasis)
        pdata=[[molecule],[nbasis]]
        #exit(0)

        #dataset = TencentAlchemyDataset(mode='valid',rootdir=path,chemspace=self.chemspace,tra_size=tra_size)
        dataset=TADataset(mode='test',rootdir=path,suits=molecule,chemspace=chemspace,pdata=pdata,tra_size=tra_size,target=self.target)
        loader = DataLoader(dataset     = dataset,
                            batch_size  = self.config.batch_size,
                            collate_fn  = batcher(),
                            shuffle     = False,
                            num_workers = 0)

        if not os.path.exists(modelname):
            print(modelname+" does not exist!")
            return
        model = th.load(modelname,map_location=self.device)
        #model.to(self.device)

        #loss_fn = nn.MSELoss()
        #MAE_fn = nn.L1Loss() 
        model.eval()
        bnums   = []
        times   = []
        preds   = []
        # sdflist = []

        with th.no_grad():
            err=0
            errs=[]
            j=0
            mae=0.0
            #with open(tmp1,'r') as ftmp:
            #    ftmplines=ftmp.readlines()
            #print(ftmplines)   

            for idx,batch in enumerate(loader):
                batch.graph=batch.graph.to(self.device)
                batch.label = batch.label.to(self.device)
                batch.basisnum=batch.basisnum.to(self.device)
                #batch.sdf=batch.basisnum.to(self.device)
                res = model(batch.graph,batch.basisnum)
                res=res.to('cpu')
                #mae = MAE_fn(res, batch.label)
                #w_mae += mae.detach().item()
                reslist=res.numpy()
                #print(reslist)
                reslist=res.tolist()
                batch.label=batch.label.to('cpu')
                batch.basisnum=batch.basisnum.to('cpu')
                timelist=batch.label.numpy()
                #print(timelist)
                timelist=timelist.tolist()
                bnumlist=batch.basisnum.numpy().tolist()

                for i in range(len(reslist)):
                    #print(i, " ===> initial < === ", reslist[i]) 
                    reslist[i]=reslist[i]
                    time=timelist[i][0]
                    ares=reslist[i]
                    #bnum=basisnums2[i]
                    bnum=bnumlist[i][0]
                       
                    #print(bnum)
                    times.append(time)
                    preds.append(ares)
                    bnums.append(bnum)
                    #sdflist.append(sdf) 
                    err1=(float(time)-float(ares))/float(time)
                    #print('i: ',i, ' sdf/mol: ', (ftmplines[i].split()[4]) ,' basis num: ',bnum,' real time : ',time,' predicted time: ',ares, 'err', err1)
                    ae=abs(time-ares)
                    single_err=ae/time
                    err+=single_err
                    mae+=ae
                    errs.append(single_err)
                    j+=1
            # err_mean=err/j
            # mae/=j
            # errs=np.array(errs)
            # variance=errs.var()
            # print("MRE：%.4f, MAE: %.4F, VAR: %.4f"%(err_mean,mae,variance))

        if write==True:
            wbname='results_'+self.chemspace+'/'+'tes_mpnn_'+str(tra_size)+'.xlsx'
            workbook=xlsxwriter.Workbook(wbname,{'constant_memory': True})# benchmark results
            worksheet=workbook.add_worksheet("")
            worksheet.write_row('A1',['nbasis','res','real'])
            for i in range(len(bnums)):
                content=[bnums[i],preds[i],times[i]]
                row='A'+str(i+2)
                worksheet.write_row(row,content)
            workbook.close()
            print("writing to xlsx done!")

        return reslist

#        return [err_mean,mae,variance]    

