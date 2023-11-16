mutable struct FRAGS
      idx::Int
     name::String
     node::String
 nodeidx0::Int
 nodeidx1::Int
 nodeidx2::Int
  timeCPU::Float64
  timeELP::Float64
end


# Evaluate the therical LB 
# e.g.
#       julia  Timing_analysis_theory.jl    Opted_Tasks.txt    LB-LSTM_P38_50_631pgs-augV 
#                                           [reserved data]    [New LB file ]

ntaskmax=29 # for avoiding GNUPLOT disorder 

taskda=ARGS[1]
target=ARGS[2]

if !isfile(taskda)
    println("The task data file $(taskda) is not exist")
    exit("Stopped. Reason: $(taskda) is not exist.")
else
    println("target task data file : ", taskda )
end

if !isfile(target)
    println("The target LB file $(target) is not exist")
    exit("Stopped. Reason: $(target) is not exist.")
else
    println("target LB file : ", target )
end

idx = 0
fraglist = []
open(taskda,"r") do readda
    while !eof(readda)
        global idx = idx + 1

	line=readline(readda)
        sline=split(line)
#	println(sline) 

	ifrag = parse(Int32,split(split(sline[4],"(")[2],",")[1])
	name  = split(split(sline[5])[1],"\"")[2]
	tCPU  = parse(Float64,split(split(sline[10])[1],",")[1])
	tELP  = parse(Float64,split(split(sline[11])[1],")")[1])

#	println(tELP) 

	push!(fraglist,FRAGS(ifrag,name,"",0,0,0,tCPU,tELP))
    end 	    
end 

#println(fraglist)
#println("debugging")
#exit(0)

tasklist=[]
open(target,"r") do readta
    line  = readline(readta)
    sline = split(line)
    nnode = parse(Int32,sline[2])
    for i in 1:nnode
        list=[] 
	line  = readline(readta)
	sline = split(line)
	for j in 1:length(sline) 
	    push!(list,parse(Int32,sline[j]))
	end
        #println(list)
	#exit(0) 
	push!(tasklist,list)
    end
end

println(tasklist[1])

open("CUSTOM-LSTM-GNUPLOT-ELP-2.dat","w") do record
print(record,"node ")	
for i in 1:length(tasklist[1])
    print(record," task",i)
end
println(record)
for i in 1:length(tasklist)
    print(record,"imnode",i)	
    ntask = 0      
    for j in 1:length(tasklist[i])
        for k in 1:length(fraglist)
            if fraglist[k].idx == tasklist[i][j]
   	        print(record," ",fraglist[k].timeELP)   
	        ntask = ntask + 1
		break 
	    end  
        end
    end
    nadd = ntaskmax - ntask
    for k in 1:nadd
        print(record," 0.0")
    end
    println(record)	
end
end


