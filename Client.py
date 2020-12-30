import sys,getopt
import socket

def main(argv):
    usage_str='''example: python Fcst_kernel_A1.py -f|--func <functional> -b|--basis <basis> -i|--input <inputfile> -m|--model <model> -n|--ncores <ncores> -c|--cpu'''
    try:
        opts,args=getopt.getopt(argv[1:],
        "hcf:b:i:m:n:d:q:",
        ["help","cpu","func=","basis=","input=","model=","ncores=","nnodes=","freq="])
    except getopt.GetoptError:
        print(usage_str)
        return
    
    msg_list=["none" for i in range(8)]
    msg_list.append("END")
    for opt,arg in opts:
        if opt in ("-h","--help"):
            print(usage_str)
            return
        elif opt in ("-c","--cpu"):
            msg_list[0]="cpu"
        elif opt in ("-f","--func"):
            msg_list[1]=arg
        elif opt in ("-b","--basis"):
            msg_list[2]=arg
        elif opt in ("-i","--input"):
            msg_list[3]=arg
        elif opt in ("-m","--model"):
            msg_list[4]=arg
        elif opt in ("-n","--ncores"):
            msg_list[5]=arg
        elif opt in ("-d","--nnodes"):
            msg_list[6]=arg
        elif opt in ("-q","--freq"):
            msg_list[7]=arg
    msg=",".join(msg_list)
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(('127.0.0.1',10001))
    s.send(msg.encode('utf-8'))
    time=s.recv(1024).decode('utf-8')
    return time

if __name__=="__main__":
    res=main(sys.argv)
    print(res)