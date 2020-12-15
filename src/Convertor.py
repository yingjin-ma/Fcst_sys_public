import openbabel


def GjfToXYZ(infile):
    if infile.split(".")[-1]!="gjf":
        raise Exception("invalid input format "+infile.split(".")[-1])
    gjf_register=[]
    with open(infile,"r") as inf:
        for line in inf.readlines():
            gjf_register.append(line)
    title=gjf_register[2].strip()
    del gjf_register[:5]
    num_atoms=len(gjf_register)-1
    outfile=infile.split("gjf")[0]+"xyz"
    with open(outfile,"w") as outf:
        outf.write(str(num_atoms)+"\n")
        outf.write(title+"\n")
        for item in gjf_register[:-1]:
            outf.write(item)
    return outfile

def GjfToSdf(infile):
    try:
        xyzfile=GjfToXYZ(infile)
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        inmol=openbabel.OBMol()
        try:
            outfile=xyzfile.split("xyz")[0]+"sdf"
            obConversion.ReadFile(inmol, xyzfile)
            obConversion.WriteFile(inmol,outfile)
            return outfile
        except:
            raise Exception("open babel conversion error")
    except:
        raise






if __name__=="__main__":
    GjfToXYZ("example/mv.gjf")
