import torch
import torch.nn as nn


# 前馈神经网络
class FNN(nn.Module):
    def __init__(self,hidden_dim,output_dim,num_layers):
        '''
        构造方法
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: 隐藏层数目
        '''
        super().__init__()
        self.input_dim=12
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers #隐藏层的数目
        self.hidden_layers=[]
        #self.bn1=nn.BatchNorm1d(1)
        self.layers=nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), 
            nn.BatchNorm1d(self.hidden_dim), 
            nn.Tanh())
        for i in range(self.num_layers):
            self.layers.add_module('h'+str(i),nn.Linear(self.hidden_dim,self.hidden_dim))
            self.layers.add_module('bn'+str(i),nn.BatchNorm1d(self.hidden_dim))
            self.layers.add_module('ac'+str(i),nn.Tanh())
        
        self.out=nn.Linear(self.hidden_dim,self.output_dim)


    def forward(self,indata):
        output1=self.layers(indata)
        output2=self.out(output1)
        return output2.squeeze(-1)

            
            