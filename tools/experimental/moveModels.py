import os
from shutil import copyfile

PREFIX="C:/codes/Fcst_sys/model_all/"

funcs=['B3LYP','bhandhlyp','BLYP','CAM-B3LYP','LC-BLYP','M06','M062x','PBE1PBE','wb97xd']
bases=['6-31g','6-31gs','6-31pgs']


rootdir=PREFIX+"Gaussian_TDDFT/"

for dir in os.listdir(rootdir):
    path=os.path.join(rootdir,dir)
    #os.mkdir(os.path.join(path,"rdmodel_ave"))
    rfdir=os.path.join(path,"rfmodel_tot")
    if not os.path.exists(rfdir):
        os.mkdir(rfdir)
    #chemspace=dir.split('-')[1]+'-'+dir.split('-')[2]
    chemspace=dir.replace('Gaussian-','')
    source=PREFIX+'Excited_states_GAUSSIAN/Gaussian-'+chemspace+'_TD-rfmodel_tot'
    for md in os.listdir(source):
        target=PREFIX+'Gaussian_TDDFT/Gaussian-'+chemspace+'/'+'rfmodel_tot'+'/'+md
        copyfile(source+'/'+md,target)
