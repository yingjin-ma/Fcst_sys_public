from pickle import GLOBAL
import torch
import Globals
import joblib

Functionals=("B3LYP","bhandhlyp","BLYP","CAM-B3LYP","LC-BLYP","M06","M062x","PBE1PBE","wb97xd")
Basis_sets=("6-31g","6-31gs","6-31pgs")
Mod_types=("rf","lstm","mpnn","mgcn")
Qc_packages=("G09",)
Machines=("ERA",)



def _init(BAK):
    global _mods_dict
    _mods_dict={}
    device=torch.device("cpu")
    if Globals.get_value("cpu")==False:
        device=torch.device("cuda:0")
    for pack in Qc_packages:
        for mach in Machines:
            for func in Functionals:
                for basis in Basis_sets:
                    mod_dir=BAK+"/"+pack+"_"+mach+"_"+func+"_"+basis+"/"
                    for mod in Mod_types:
                        if mod=="rf":
                            mod_dir+="rfmodel_tot/"
                            for i in range(1,5):
                                mod_name=func+"_"+basis+"_"+str(i)+".pkl"
                                mod=torch.load(mod_name,map_location=device)
                                _mods_dict[mod_name]=mod
                            _mods_dict["RFC"]=joblib.load(BAK+"/RFC.m")
                        else:
                            mod_name=mod_dir+mod+"_"+func+"_"+basis+"_tot.pkl"
                            mod=torch.load(mod_name,map_location=device)
                            _mods_dict[mod_name]=mod


def mod_exists(mod_name):
    return mod_name in _mods_dict.keys()

def getModel(mod_loc):
    return _mods_dict[mod_loc]


