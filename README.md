# A Forecasting System under the Multiverse ansatz via Machine Learning and Cheminformatics

## Specific scope : Computational Time of DFT/TDDFT Calculations for this version

### 1. Catalogue
  - database : 
    - DrugBank : DrugBank all-in-one sdf file, and the scripts for generating the training and the testing suits
    - polyfitted : Fitted ploynormal equations for selected 89 reference DFT functionals
    - rawdata : The assembled Gaussuian09-D.01 timing data, and the separated sdf files with added H atoms
    - trained-models : Trained models for few DFT functional/basis set combinations
  - example : The sample molecule to be predicted 
  - src : source code folder
  - tools : Independent scripts 
    - experimental : Some scripts in developing or experimental stage
  - TRmod_kernel_A1.py : Training script sample
  - Fcst_kernel_A1.py : Predicting script sample

### 2. Installation
  - Prerequisities
    - python3 with numpy, scipy, scikit-learn
    - pytorch, with CUDA, cudatoolkit, torchvision, dgl, gensim 
    - basis_set_exchange, libxc
    - rdkit, openbabel
    - optional: xlsxwriter, pillow
  - Installation example (recommended with conda):
    - git clone git@github.com:yingjin-ma/Fcst_sys_public.git Fcst_sys_public
    - cd Fcst_sys_public
    - conda create -n Fcst_sys_public python=3.7
    - conda activate Fcst_sys_public
    - conda install rdkit pytorch gensim torchvision numpy scipy xlsxwriter scikit-learn basis_set_exchange libxc matplotlib
    - pip install dgl-cu100  (notice the cu version should match that of cudatoolkit)
      - *now: python TRmod_kernel_A1.py should work*
    - Install the pylibxc (Please see https://www.tddft.org/programs/libxc/installation/)
      - *now: python Fcst_kernel_A1.py should work*

### 3. Usage
    - to be updated
    
### 4. Citation
    - https://arxiv.org/abs/1911.05569v2

### 5. Corresponding authors
    - yingjin_ma@163.com or yingjin.ma@sccas.cn
    - zjin@sccas.cn



