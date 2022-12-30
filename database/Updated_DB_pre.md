# 计算结果汇总 

## P38蛋白质残基体系
### 单点计算1
    计算软件         Gaussian09
    计算参数         #p MP2=FullDirect 6-31G** sp
    结果所在文件夹    /work1/scquant/ParaEngine-git-QCs/Test_P38-MP2-631gss-fulldirect 里面，一系列机器名开头的文件夹
    结果文件         .log文件
    
## SARS-CoV2 hACE2 体系 
### 单点计算1 
    计算软件         Gaussian09
    计算参数         #p M062x/6-31G* sp
    结果所在文件夹    /work1/scquant/ParaEngine-git-QCs/Test-ACE2_Ab-Omicron_Q493R_8_4-cs-STD-DLB-50nodes 里面，一系列机器名开头的文件夹
    结果文件         .log文件
### 单点计算2（大批量）
    计算软件         Gaussian09
    计算参数         #p M062X/6-31G** scf(maxcyc=100) empiricaldispersion=GD3
    结果所在文件夹    /work1/scquant/ParaEngine-git-COVID2019/Results_M062x_631Gss-GD3-12A 文件夹，内有不同的对接结构文件夹（约80个左右），每个对接结构文件夹下面一系列机器名开头的文件夹
                    /work1/scquant/ParaEngine-git-COVID2019/Results_wb97xd-631Gss-12A/Workdir-631gs-ACE2_Ab-Omicron_Q493R_X_Y(X从0到9,Y从0到9，最后一个是9_6)
    结果文件         .log文件
   
## 药物小分子体系
### 结构优化
    计算软件         Gaussian09
    计算参数         #p B3LYP/6-31G(d) opt
    结果所在文件夹    /work1/scquant/ParaEngine-git-Drug_small_molecule/Test-test100-opt/opt-log(88个)
    结果文件         .log文件       
### 红外、拉曼波谱1
    计算软件         Gaussian09
    计算参数         #p B3LYP/6-31G(d) IOp(7/64=3) freq=raman pop=full
    结果所在文件夹    /work1/scquant/ParaEngine-git-Drug_small_molecule/Test-test100-freq/freqs(88个)
    结果文件         .log文件        
### 紫外可见、圆二色谱1
    计算软件         Gaussian09
    计算参数         #p td B3LYP/6-31G(d)
    结果所在文件夹    /work1/scquant/ParaEngine-git-Drug_small_molecule/Test-test100-td/tds(88个)
    结果文件         .log文件  
    
