import openbabel
import re



def GjfParser(infile):
    if infile.split(".")[-1]!="gjf":
        raise Exception("invalid input format "+infile.split(".")[-1])
    patterns=[]
    flag=False
    patterns[0]=re.compile("nprocs?=(\d+)") # number of cores
    patterns[1]=re.compile("nprocl=(\d+)") # number of nodes
    patterns[2]=re.compile("#[a-zA-Z]\s([a-zA-Z0-9\+\*-]+)/([a-zA-Z0-9\+\*-]+)") #functional and basis set
    patterns[3]=re.compile("[a-zA-Z]+\s+-?[0-9]+\.?[0-9]+\s+-?[0-9]+\.?[0-9]+\s+-?[0-9]+\.?[0-9]+")
    results={"nproc":24,"nprocl":1,"func":"","basis":"","coords":[]} # nproc,nprocl,func,basis,coordinates
    with open(infile,"r") as inf:
        for line in inf.readlines():
            if flag==False:
                shobj=re.search(patterns[0],line)
                if shobj is not None:
                    results["nproc"]=int(shobj.group())
                    continue
                shobj=re.search(patterns[1],line)
                if shobj is not None:
                    results["nprocl"]=int(shobj.group())
                    continue
                shobj=re.search(patterns[2],line)
                if shobj is not None:
                    results["func"]=shobj.group(1)
                    results["basis"]=shobj.group(2)
                    flag=True
            else:
                shobj=re.match(patterns[3],line)
                if shobj is not None:
                    results["coords"].append(line)
    return results    


def GjfToXyz(infile):
    try:
        results=GjfParser(infile)
        num_atoms=len(results["coords"])
        title=infile.split(".gjf")[0]
        outfile=infile.split("gjf")[0]+"xyz"
        with open(outfile,"w") as outf:
            outf.write(str(num_atoms)+"\n")
            outf.write(title+"\n")
            for item in results["coords"]:
                outf.write(item)
        return outfile,results
    except:
        raise

# def GjfToXyz(infile):
#     if infile.split(".")[-1]!="gjf":
#         raise Exception("invalid input format "+infile.split(".")[-1])
#     gjf_register=[]
#     with open(infile,"r") as inf:
#         for line in inf.readlines():
#             gjf_register.append(line)
#     title=gjf_register[2].strip()
#     del gjf_register[:5]
#     num_atoms=len(gjf_register)-1
#     outfile=infile.split("gjf")[0]+"xyz"
#     with open(outfile,"w") as outf:
#         outf.write(str(num_atoms)+"\n")
#         outf.write(title+"\n")
#         for item in gjf_register[:-1]:
#             outf.write(item)
#     return outfile

def GjfToSdf(infile):
    try:
        xyzfile,results=GjfToXyz(infile)
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        inmol=openbabel.OBMol()
        try:
            outfile=xyzfile.split("xyz")[0]+"sdf"
            obConversion.ReadFile(inmol, xyzfile)
            obConversion.WriteFile(inmol,outfile)
            return outfile,results
        except:
            raise Exception("open babel conversion error")
    except:
        raise







if __name__=="__main__":
    GjfToXYZ("example/mv.gjf")
