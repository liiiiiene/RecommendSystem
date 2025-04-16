import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self,norm_shape,dropout,max_len=10000,**kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros(size=(1,max_len,norm_shape))
        x = torch.arange(max_len).reshape(-1,1) / torch.pow(10000,torch.arange(1,norm_shape+1,2,dtype=torch.float32) / norm_shape)
        self.P[0,:,0::2] = torch.sin(x)
        self.P[0,:,1::2] = torch.cos(x)
    def forward(self,X):
        X = X + self.P[:,X.shape[1],:].to(X.device)
        return self.dropout(X)
    
    