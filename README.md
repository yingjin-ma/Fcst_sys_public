# A Forecasting System under the Multiverse ansatz via Machine Learning and Cheminformatics

## Specific scope : Computational Time of DFT/TDDFT Calculations for this version
  - Thank PARATERA company (https://www.paratera.com/) for the cooperations

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
  - *python TRmod_kernel_A1.py* for training
  - *python Fcst_kernel_A1.py* for predicting
  - *python Fcst_kernel_A1_LB_wrapper.py* for load-balancing
    - *The "Predicted_Loads.txt" will be generated for later usage*
  - *More will be added*
    
### 4. Citation
  - https://arxiv.org/abs/1911.05569v3
  - https://pubs.acs.org/doi/10.1021/acsomega.0c04981

### 5. Acknowledgement
  - National Key Research and Development Program of China (Grant No.2018YFB0203805)
  - National Natural Science Foundation of China (Grant No.21703260)
  - Informationization Program of the Chinese Academy of Science (Grant No.XXH13506-403)
  - Guangdong Provincial Key Laboratory of Biocomputing (Grant No.2016B030301007)

### 6. Corresponding authors
  - yingjin_ma@163.com and zjin@sccas.cn



