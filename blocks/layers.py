import torch.nn as nn 
import numpy as np 

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding,self).__init__()    
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]  

class ResidualCennection(nn.Module):
    def __init__(self,x, residual):
        super(ResidualCennection,self).__init__()
        self.pass_trough = x
        self.addtion = residual
    def forward(self,*args, **kwargs):
        x = self.pass_trough
        return x + self.addtion


class FeedForwrdNetwork(nn.Module):
    def __init__(self,embed_size , hidden_size , dropout_pro=0.1):
        super(FeedForwrdNetwork,self).__init__()
        self.Linear_1= nn.Linear(embed_size,hidden_size)
        self.Linear_2= nn.Linear(hidden_size,embed_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_pro)
    def forward(self,x):
        x = self.Linear_1(x)
        x = self.act(x)
        x = self.Linear_2(x)
        x = self.dropout(x)
        return x

class NormLayer(nn.Module):
    def __init__(self,embed_size , eps=1e-12):
        super(NormLayer, self).__init__()
        self.embed_size = embed_size
        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1, keepdims= True)
        var = x.var(-1 , unbiased=False , keepdims= True)
        out = ( x - mean ) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out




