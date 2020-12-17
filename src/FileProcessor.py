import openbabel
import re
import getopt
import sys



def GjfParser(infile):
    if infile.split(".")[-1]!="gjf":
        raise Exception("invalid input format "+infile.split(".")[-1])
    
    flag=False
    pattern0=re.compile("[0-9]\s+[0-9]")
    pattern1=re.compile("nprocs?=(\d+)") # number of cores
    pattern2=re.compile("nprocl=(\d+)") # number of nodes
    pattern3=re.compile("#[a-zA-Z]\s([a-zA-Z0-9\+\*-]+)/([a-zA-Z0-9\+\*-]+)") #functional and basis set
    pattern4=re.compile("[a-zA-Z]+\s+-?[0-9]+\.?[0-9]+\s+-?[0-9]+\.?[0-9]+\s+-?[0-9]+\.?[0-9]+")
    results={"nproc":24,"nprocl":1,"func":"","basis":"","coords":[]} # nproc,nprocl,func,basis,coordinates
    with open(infile,"r") as inf:
        for line in inf.readlines():
            if flag==False:
                shobj=re.match(pattern0,line)
                if shobj is not None:
                    flag=True
                    continue
                shobj=re.search(pattern1,line)
                if shobj is not None:
                    results["nproc"]=int(shobj.group(1))
                    continue
                shobj=re.search(pattern2,line)
                if shobj is not None:
                    results["nprocl"]=int(shobj.group(1))
                    continue
                shobj=re.search(pattern3,line)
                if shobj is not None:
                    results["func"]=shobj.group(1)
                    results["basis"]=shobj.group(2)
                    flag=True
            else:
                shobj=re.match(pattern4,line)
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


def main(argv):
    try:
        opts,args=getopt.getopt(argv[1:],"i:",["input="])
    except getopt.GetoptError:
        exit()
    for opt,arg in opts:
        if opt in ("-i","--input"):
            name,results=GjfToSdf(arg)
            print("basis set: ",results["basis"])
            print("functional: ",results["func"])
        




if __name__=="__main__":
    #GjfToXYZ("example/mv.gjf")
    main(sys.argv)
