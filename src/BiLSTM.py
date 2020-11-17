import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#双向LSTM神经网络
class BiLSTM(nn.Module):
    def __init__(self,embed_dim,hidden_dim,output_dim,num_layers,weight):
        '''
        构造方法
        embed_dim: 输入特征维度,即每个词向量的维度
        hidden_dim: 隐状态的维度
        output_dim: 输出状态的维度（根据smiles编码所提取的特征的维度）
        num_layers: LSTM隐藏层的层数，是这么多个LSTM的串联,不同于时序展开
        weight: 每个词的权重组成的集合，由预训练的词向量得到
        '''
        super().__init__()
        self.input_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.biFlag=True
        self.embedding=nn.Embedding(74,embed_dim)
        #self.embedding=nn.Embedding.from_pretrained(weight)
        #self.embedding.weight.requires_grad=False
        #LSTM作为特征提取器 input shape:[batchsize,seqlen,embedsize]
        self.lstm=nn.LSTM(input_size=embed_dim,hidden_size=self.hidden_dim,
            num_layers=num_layers,batch_first=True,bidirectional=self.biFlag,dropout=0.5)  #final output shape: [batchsize,maxtime=maxlen=seqlen,hidden_dim*2]
        #attention层
        self.atten_layer=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.relu=nn.ReLU(inplace=True)
        self.tanh=nn.Tanh()
        self.decoder=nn.Linear(hidden_dim,output_dim)
        self.bn=nn.BatchNorm1d(output_dim+1)#批标准化
        self.bn2=nn.BatchNorm1d(10)
        #5个全连接层，用于拟合
        self.fc1=nn.Linear(output_dim+1,10)
        self.fc2=nn.Linear(10,10)
        self.fc3=nn.Linear(10,10)
        self.fc4=nn.Linear(10,5)
        self.fc5=nn.Linear(5,1) 

    def attention(self,lstm_out):
        #lstm_out shape: [batchsize,time_step,hidden_dim*2]
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)#将lstm_out分块，沿着最后一个维度分为2块
        # [batch_size, time_step, hidden_dim]
        h=lstm_tmp_out[0]+lstm_tmp_out[1] #将前向LSTM和后向LSTM的输出相加 [batch_size, time_step, hidden_dim]
        atten_w=self.atten_layer(h) # [batch_size, time_step, hidden_dim]
        m=nn.Tanh()(h) #[batch_size, time_step, hidden_dim]
        atten_context=torch.bmm(m,atten_w.transpose(1,2)) #[batch_size, time_step, time_step]
        softmax_w=F.softmax(atten_context,-1)
        context=torch.bmm(h.transpose(1,2),softmax_w) #[batch_size, hidden_dim, time_step]
        result=torch.sum(context,dim=2) # [batch_size,hidden_dim]
        #result=torch.squeeze(result,2) # [batch_size,hidden_dim]
        result=nn.Dropout()(result)
        return result

    def forward(self,smiles,basisnums):
        #smiles [batch_size, seq_len]  basisnums [batch_size]
        embeddings=self.embedding(smiles) #embeddings [batch_size, seq_len, embedsize]
        states , _ = self.lstm(embeddings) #states [batch_size, seq_len, hidden_dim*2]
        atten_out=self.attention(states) #atten_out [batch_size,hidden_dim]
        decoder_out=self.decoder(atten_out) # [batch_size,output_dim]
        basisnums=torch.unsqueeze(basisnums,1)
        inputbn=torch.cat((decoder_out,basisnums),1)
        #print(inputbn)
        inputfc1=self.bn(inputbn)
        #print(inputfc1)
        outputfc1=self.relu(self.fc1(inputfc1))
        outputfc2=self.relu(self.fc2(outputfc1))
        outputfc3=self.tanh(self.fc3(outputfc2))
        outputfc3=self.bn2(outputfc3)
        outputfc4=self.fc4(outputfc3)
        outputfc5=self.fc5(outputfc4)
        return outputfc5.squeeze(-1)


    
