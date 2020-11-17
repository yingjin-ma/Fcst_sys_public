import sys
import joblib

PREFIX="/data1/Fcst_sys"
sys.path.append(PREFIX)


import LstmTool4
import RfTool4
import MpnnTool4
import MgcnTool4

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import Configs

torch.manual_seed(2)

rootDir=PREFIX+'/model_all/Gaussian_ground/'


def eval_rf(chemspace,folder_sdf,aim,valids):
    spaceDir=rootDir+"Gaussian-"+chemspace+"/"
    aimming=2
    modelDir=spaceDir+"rfmodel_tot"
    if aim=='ave':
        aimming=6
        modelDir=spaceDir+"rfmodel_ave"
    
    training_size=500
    series=['alkane','branch','ring','PE']
    config=Configs.Config(tra_num_epochs=3000,tra_size=training_size,
    lr=0.005,batch_size=40,tra_set_ratio=1,valid_interval=2)
    tool=RfTool4.RfTool(chemspace=chemspace,config=config,sdf_dir=folder_sdf[0],target=aimming)
    tool.clf = joblib.load(PREFIX+"/model/RFC.m")
    data_each=tool.eval_each(path=valids,modeldir=modelDir)
    return data_each


def eval_lstm(chemspace,folder_sdf,suits_train,suits_valid,aim):
    spaceDir=rootDir+"Gaussian-"+chemspace+"/"
    aimming=2
    suffix='tot'
    if aim=='ave':
        aimming=6
        suffix='ave'
    training_size=2000
    config=Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,
    batch_size=100,tra_set_ratio=1,valid_interval=2)
    modelName=spaceDir+'lstm_'+chemspace+'_'+suffix+'.pkl'
    tool=LstmTool4.LstmTool(chemspace=chemspace,config=config,
    suits1=suits_train,suits2=suits_valid,suits3=[],
    modelloc=modelName,sdf_dir=folder_sdf[0], target=aimming)
    data_each=tool.eval_each(modelname=modelName,path=suits_valid,chemspace=chemspace)
    return data_each

def eval_mpnn(chemspace,folder_sdf,suits_train,suits_valid,aim):
    spaceDir=rootDir+"Gaussian-"+chemspace+"/"
    aimming=2
    suffix='tot'
    if aim=='ave':
        aimming=6
        suffix='ave'
    training_size=2000
    config=Configs.Config(tra_num_epochs=600,tra_size=training_size,lr=0.002,
    batch_size=100,tra_set_ratio=1,valid_interval=2)
    modelName=spaceDir+'mpnn_'+chemspace+'_'+suffix+'.pkl'
    tool=MpnnTool4.MpnnTool(chemspace=chemspace,config=config,
    suits1=suits_train,suits2=suits_valid,suits3=[],folder_sdf=folder_sdf,
    target=aimming)
    data_each=tool.eval_each(modelname=modelName,chemspace=chemspace,path=spaceDir)
    return data_each

def eval_mgcn(chemspace,folder_sdf,suits_train,suits_valid,aim):
    spaceDir=rootDir+"Gaussian-"+chemspace+"/"
    aimming=2
    suffix='tot'
    if aim=='ave':
        aimming=6
        suffix='ave'
    training_size=2000
    config=Configs.Config(tra_num_epochs=500,tra_size=training_size,lr=0.005,
    batch_size=100,tra_set_ratio=1,valid_interval=2)
    modelName=spaceDir+'mgcn_'+chemspace+'_'+suffix+'.pkl'
    tool=MgcnTool4.MgcnTool(chemspace=chemspace,config=config,suits1=suits_train,
    suits2=suits_valid,suits3=[],folder_sdf=folder_sdf,target=aimming)
    data_each=tool.eval_each(modelname=modelName,chemspace=chemspace,path=spaceDir)
    return data_each



#if __name__=='main':
bases=['6-31gs','6-31pgs']
funcs=['M062x']

folder_sdf=[]
folder_sdf.append(PREFIX+"/16-2-2_1to25_M062x-631g/sdf_only_valid_atoms_working5_with_H")


for func in funcs:
    for basis in bases:
        chemspace=func+'_'+basis
        suits_train=[]
        dataDir=PREFIX+'/Spaces/Gaussian-'+chemspace+'/data/'
        suits_train.append(dataDir+"branch")
        suits_train.append(dataDir+"ring")
        suits_train.append(dataDir+"ring_sub")
        suits_train.append(dataDir+"alkane")
        suits_train.append(dataDir+"PE")
        suits_train.append(dataDir+"Gaussian_inputs_training2")
        suits_train.append(dataDir+"Gaussian_inputs_training")
        suits_train.append(dataDir+"Gaussian_inputs_training3")
        suits_train.append(dataDir+"Gaussian_inputs_training4")
        suits_train.append(dataDir+"Gaussian_inputs_training5")

        suits_valid=[]
        suits_valid.append(dataDir+"Gaussian_inputs_validing")
        suits_valid.append(dataDir+"Gaussian_inputs_validing2")
        suits_valid.append(dataDir+"Gaussian_inputs_testing")
        suits_valid.append(dataDir+"Gaussian_inputs_testing2")
        res_rf=eval_rf(chemspace,folder_sdf,aim='tot',valids=suits_valid)
        res_lstm=eval_lstm(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        res_mpnn=eval_mpnn(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        res_mgcn=eval_mgcn(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        with open(PREFIX+'/model_all/rf_'+chemspace+'.log','a+') as log:
            for item in res_rf:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)
        with open(PREFIX+'/model_all/lstm_'+chemspace+'.log','a+') as log:
            for item in res_lstm:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)        
        with open(PREFIX+'/model_all/mpnn_'+chemspace+'.log','a+') as log:
            for item in res_mpnn:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)
        with open(PREFIX+'/model_all/mgcn_'+chemspace+'.log','a+') as log:
            for item in res_mgcn:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)


for func in funcs:
    for basis in bases:
        rootDir=PREFIX+'/model_all/Gaussian_TDDFT/'
        chemspace=func+'_'+basis
        suits_train=[]
        dataDir=PREFIX+'/Spaces_others/Excited_states_GAUSSIAN/Gaussian-'+chemspace+'_TD-'
        suits_train.append(dataDir+"branch")
        suits_train.append(dataDir+"ring")
        suits_train.append(dataDir+"ring_sub")
        suits_train.append(dataDir+"alkane")
        suits_train.append(dataDir+"PE")
        suits_train.append(dataDir+"Gaussian_inputs_training2")
        suits_train.append(dataDir+"Gaussian_inputs_training")
        suits_train.append(dataDir+"Gaussian_inputs_training3")
        suits_train.append(dataDir+"Gaussian_inputs_training4")
        suits_train.append(dataDir+"Gaussian_inputs_training5")

        suits_valid=[]
        suits_valid.append(dataDir+"Gaussian_inputs_validing")
        suits_valid.append(dataDir+"Gaussian_inputs_validing2")
        suits_valid.append(dataDir+"Gaussian_inputs_testing")
        suits_valid.append(dataDir+"Gaussian_inputs_testing2")
        res_rf=eval_rf(chemspace,folder_sdf,aim='tot',valids=suits_valid)
        res_lstm=eval_lstm(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        res_mpnn=eval_mpnn(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        res_mgcn=eval_mgcn(chemspace,folder_sdf,suits_train,suits_valid,'tot')
        with open(PREFIX+'/model_all/TD_rf_'+chemspace+'.log','a+') as log:
            for item in res_rf:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)
        with open(PREFIX+'/model_all/TD_lstm_'+chemspace+'.log','a+') as log:
            for item in res_lstm:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)        
        with open(PREFIX+'/model_all/TD_mpnn_'+chemspace+'.log','a+') as log:
            for item in res_mpnn:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)
        with open(PREFIX+'/model_all/TD_mgcn_'+chemspace+'.log','a+') as log:
            for item in res_mgcn:
                line=str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n'
                log.write(line)
