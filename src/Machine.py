
# queue info
queues={'cpu-1':['zjin'],'cpu-2':['c_soft']}

# calculate the speed rate of E5-2680V3 and the target cpu 
def GetSpeedRate(ncores,queue,ref_speed):
    turbo_freq_maxcores=2.9
    if queue in queues['cpu-1']:
        turbo_freq_maxcores=3.1
    speed=ncores*(1.015-ncores*0.0067)*turbo_freq_maxcores
    return ref_speed/speed