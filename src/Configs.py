import os
import sys
# Config.py

#此类用于设置模型的超参数
class Config:

    def __init__(self,tra_num_epochs=15,input_size=100,
        hidden_dim=100,num_layers=2,output_dim=5,batch_size=128,lr=0.01,tra_size=4000,tra_set_ratio=0.75,valid_interval=5):
        '''
        构造方法，一般情况下仅需指定tra_num_epochs、batch_size、lr三个参数；
        tra_num_epochs=25: 训练时的epoch数目
        input_size=100: 模型输入维度，仅对LSTM生效
        hidden_dim=100: 隐藏层维度，仅对RF和LSTM生效
        num_layers=2: 用于指定nn.LSTM串联的数目
        output_dim=5: 用于指定LSTM的输出维度
        batch_size=128: 每个batch的大小，若使用GPU训练，经测试4G显存下对于MPNN该值应设置为32
        lr=0.001: 学习率 
        tra_size=4000: 训练集规模，现阶段仅允许设置为4000、2000、1000、500，除非要测试训练集规模对模型的影响，否则不用改动
        tra_set_ratio=0.75: 训练集大小/(训练集大小+验证集大小)
        valid_interval=5: 其值表示每跑多少轮进行一次验证
        '''
        self.tra_num_epochs=tra_num_epochs
        self.input_size=input_size
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.output_dim=output_dim
        self.batch_size=batch_size
        self.lr=lr
        self.tra_size=tra_size
        self.tra_set_ratio=tra_set_ratio
        self.valid_interval=valid_interval
   
        print("  tra_size : ",tra_size, "tra_num_epochs : ",tra_num_epochs, "  batch_size : ", batch_size, "  lr : ",lr, "  tra_set_ratio : ", tra_set_ratio, "  valid_interval : ", valid_interval)
