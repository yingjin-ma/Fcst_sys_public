# This code belongs to the ParaEngine/Fcst_sys package
#    - Application of fragment TDDFT calculation (1 excition)
#      with N-body interactions
# 

include("fragment-TDDFT_lib.jl")

# e.g. usage
# julia   fragment-TDDFT.jl  Target-PDB[XXX.pdb]   N-nody interaction[1,2,3]    

target  =             ARGS[1]
nbody   = parse(Int32,ARGS[2])

ResID      = [1]       # The excited frag
flagMODEL1 = "MODEL"
flagMODEL2 = "ENDMD"
flagADDH   = ""

if !isfile(target)
    println("The target file $(target) is not exist")
    exit("Stopped. Reason: $(target) is not exist.")
else
    println("The input PDB      : ", target )
end

if nbody != 2
    println("Only support 2-body interactions")
    exit("Stopped. Reason: 2 is valid.")
else
    println("N-body interaction : ", nbody )
end

readpdbfile(target,flagADDH)
if flagADDH == "ADDH"
    residueADDH()
    atomlist=[]
    fraglist=[]
    atomlist=atomlist1
    fraglist=fraglist1
end


