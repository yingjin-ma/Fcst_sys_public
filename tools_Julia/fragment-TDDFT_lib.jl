using Printf

mutable struct ATOMS
      idx::Int
    ifrag::Int
  icharge::Int
     elem::String
    coord::Tuple{Float64,Float64,Float64}
       ZZ::Float64
  residue::String
end

mutable struct FRAGS
      idx::Int
    iatom::Int
   natoms::Int
  icharge::Int
 multiple::Int
     name::String
   energy::Float64
end

function getmult(frag,atoms,ipdb)

    # ipdb :  1   PDB style
    # ipdb : -1   XYZ style

    elemdict=Dict( "H" => 1,
               "Li" => 3, "Be" => 4, "B" => 5, "C" => 6, "N" => 7, "O" => 8, "F" => 9,
               "P" => 15, "S" => 16, "Cl" => 17 )

    istart  = frag.iatom
    ifinish = frag.iatom + frag.natoms - 1

    #println(" istart, ifinish : ", istart, "  " , ifinish)


    nele=0
    for i in istart:ifinish
        #println(" ELEM : ",atoms[i].elem)
        if ipdb == 1
            elem = strip(atoms[i].elem[2:2])
        elseif ipdb == -1
            elem = strip(atoms[i].elem)
        end
        ielem = elemdict[elem]
        nele = nele + ielem
    end

    nele = nele - frag.icharge
    imod = mod(nele,2)

    frag.multiple = 2 * imod +1

    #println("nele, Multi", nele, frag.multiple)

end

function residueADDH()

    println("Add the H to the residues")

    i  = 0
    ii = 0
    iatom = 0
    names1 = []
    icharges1 = []
    for id in 1:length(ResID2)
        i = i + 1
        if ResID2[id] in ResID
                  ii = ii + 1
             istart1 = fraglist[i-1].iatom
            ifinish1 = fraglist[i-1].iatom+fraglist[i-1].natoms-1
             istartx = fraglist[i].iatom
            ifinishx = fraglist[i].iatom+fraglist[i].natoms-1
             istart3 = fraglist[i+1].iatom
            ifinish3 = fraglist[i+1].iatom+fraglist[i+1].natoms-1
              atoms1 = atomlist[istart1:ifinish1]
              atomsx = atomlist[istartx:ifinishx]
              atoms3 = atomlist[istart3:ifinish3]
             id1,id3,v1,v3 = minIDFRAG(atomsx,atoms1,atoms3)
            #println("addH1[",i-1,"] : ", id1, "   addH2[",i+1,"] : ", id3)

            name = fraglist[i].name
            push!(names1,name)
            push!(icharges1,fraglist[i].icharge)

            # push!(atomlist1,ATOMS(1+iatom,ii,0," H s",atoms1[id1].coord,0.0,atomsx[1].residue))
            push!(atomlist1,ATOMS(1+iatom,ii,0," H s",v1,0.0,atomsx[1].residue))
            natomsx = length(atomsx)
            println("natomsx", natomsx , " ResID2[id]",ResID2[id] )
            for j in 1:natomsx
                push!(atomlist1,ATOMS(1+j+iatom,ii,atomsx[j].icharge,atomsx[j].elem,atomsx[j].coord,0.0,atomsx[j].residue))
            end
            # push!(atomlist1,ATOMS(2+natomsx+iatom,ii,0," H f",atoms3[id3].coord,0.0,atomsx[1].residue))
            push!(atomlist1,ATOMS(2+natomsx+iatom,ii,0," H f",v3,0.0,atomsx[1].residue))
            iatom = iatom + 2 + natomsx

        end
    end
    global total_atoms1=iatom
    for i in 1:20
        if i > total_atoms1
            break
        else
            println(atomlist1[i])
        end
    end
    if total_atoms1 > 20
        println("... (more) for CAPPED residues ")
    end

    ifrag=0
    global fraglist1=[]
    icharge    = 0
    multiple   = 1
    natoms_frg = 0
    for i in 1:total_atoms1
        #println("atomlist[i][1] : ",atomlist1[i].ifrag)
        if atomlist1[i].ifrag != ifrag
            if ifrag != 0
                istart=atomlist1[i].idx-natoms_frg
                icharge=icharges1[ifrag]
                idx = parse(Int32,split(names1[ifrag],"-")[3])
                push!(fraglist1,FRAGS(idx,istart,natoms_frg,icharge,1,names1[ifrag],0.0))
            end
            ifrag = ifrag+1
            natoms_frg = 0
        end
        if atomlist1[i].ifrag == ifrag
            natoms_frg=natoms_frg+1
        end
        if atomlist1[i].icharge != 0
            #println("atomlist1[i][3] : ",atomlist1[i].icharge)
            icharge=icharge+atomlist1[i].icharge
        end
        if i == total_atoms1
            istart=atomlist1[i].idx-natoms_frg+1
            idx = parse(Int32,split(names1[ifrag],"-")[3])
            push!(fraglist1,FRAGS(idx,istart,natoms_frg,icharge,1,names1[ifrag],0.0))
        end
    end
    global total_frags1=ifrag

    println("total molecules in capped residue suit : ",total_frags1)
    for i in 1:total_frags1
        println(" fraglist1[",i,"]",fraglist1[i])
        getmult(fraglist1[i],atomlist1,1)
        #println(" ==== ")
        #println(fraglist1[i])
        #println(" ==== 2 ==== ")
    end
end


function readpdbfile(inpdb,flag)

    global atomlist1=[]
    global fraglist1=[]

    if uppercase(flag) == "ADDH"
        ResTMP=[]
        for id in 1:length(ResID)
            push!(ResTMP,ResID[id]-1)
            push!(ResTMP,ResID[id])
            push!(ResTMP,ResID[id]+1)
        end
        global ResID2 = unique(ResTMP)
    else
        ResID2 = ResID
    end

    println("ResID extended : ", ResID2)

    model=[]
    model_natoms=[]
    imodel=0 
    open(inpdb,"r") do pdbstream
        for line in eachline(pdbstream)
            sline=split(line)
            ntmp=length(sline)
            if ntmp > 0
                # One time one residual/part/fragment
                if uppercase(line[1:5])==flagMODEL1 || flagMODEL1 ==""
                    global natoms=0
                    global atomlist=[]
                end                  
                if ntmp > 10 
                        #println(line)
                        icharge= 0
                        natoms = natoms + 1
                         ifrag = parse(Int32,line[23:26])
                            dx = parse(Float64,line[31:38])
                            dy = parse(Float64,line[39:46])
                            dz = parse(Float64,line[47:54])
                        if line[79:79] != " "
                            # println("line : ",line[79:79])
                            icharge = parse(Int32,line[79:79])
                        end
                        if length(line) > 80
                            icharge = parse(Int32,line[81:82])
                            #push!(atomlist,ATOMS(natoms,ifrag,icharge,line[13:16],(dx,dy,dz),0.0,line[18:26]))
                            push!(atomlist,ATOMS(natoms,ifrag,icharge,line[77:80],(dx,dy,dz),0.0,line[18:26]))
                        else
                            if line[80:80] == "-"
                                icharge = -1 * icharge
                            end
                            #push!(atomlist,ATOMS(natoms,ifrag,icharge,line[13:16],(dx,dy,dz),0.0,line[18:26]))
                            push!(atomlist,ATOMS(natoms,ifrag,icharge,line[77:80],(dx,dy,dz),0.0,line[18:26]))
                        end
                end
                if uppercase(line[1:5])==flagMODEL2 || flagMODEL2==""
                    imodel = imodel + 1 
                    push!(model,atomlist) 
                    push!(model_natoms,natoms)
                end                  
            end
        end
    end
    #println(model)
    #println(model_natoms)
    #exit(1)

    model_nfrags=[]
    for imodel in 1:length(model)
        ifrag=1
        total_atoms = model_natoms[imodel]
        for i in 1:total_atoms-1
            idiff = -1
            if model[imodel][i].ifrag != model[imodel][i+1].ifrag
                idiff = 1
            end
            model[imodel][i].ifrag = ifrag
            if idiff == 1
                ifrag = ifrag + 1
            end
        end
        model[imodel][total_atoms].ifrag = ifrag

        if imodel == length(model)
            println(" === Print the final models for checking === ")
            for i in 1:20
                if i > total_atoms
                    break
                else
                    println(model[imodel][i])
                end
            end
            if total_atoms > 20
                println("... (more) ")
            end
        end 

        ifrag=0
        global fraglist=[]
        icharge    = 0
        multiple   = 1
        natoms_frg = 0
        for i in 1:total_atoms
            #println("model[imodel][i].ifrag : ",model[imodel][i].ifrag)
            if model[imodel][i].ifrag != ifrag
                if ifrag != 0
                    istart=model[imodel][i].idx-natoms_frg            
                    #println(model[imodel][i-1])        
                    idx = parse(Int32,split(model[imodel][i-1].residue)[2])
                    push!(fraglist,FRAGS(idx,istart,natoms_frg,icharge,1,replace(model[imodel][i-1].residue," "=>"-"),0.0))
                end
                ifrag = ifrag+1
                icharge    = 0
                natoms_frg = 0
            end
            if model[imodel][i].ifrag == ifrag
                natoms_frg=natoms_frg+1
            end
            if model[imodel][i].icharge != 0
                #println("atomlist[i][3] : ",model[imodel][i].icharge)
                icharge=icharge+model[imodel][i].icharge
            end
            if i == total_atoms
                istart=model[imodel][i].idx-natoms_frg+1
                idx = parse(Int32,split(model[imodel][i].residue)[2])
                push!(fraglist,FRAGS(idx,istart,natoms_frg,icharge,1,replace(model[imodel][i].residue," "=>"-"),0.0))
            end
        end
        global total_frags=ifrag

        if imodel == length(model)
            println("total_frags : ",total_frags)
        end 
        for i in 1:total_frags
            getmult(fraglist[i],model[imodel],1)
            #println(" ==== ")
            if imodel == length(model)
                println(fraglist[i])
            end 
        end
        push!(model_nfrags,total_frags)
    end   
end

