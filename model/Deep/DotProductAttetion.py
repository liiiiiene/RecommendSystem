import torch
from torch import nn
from torch.nn import functional as F
import math

def sequence_mask(X,valid_len,value=0):
    max_lenght = X.shape[1]
    mask = torch.arange(max_lenght,dtype=torch.float32,device=X.device)[None,:] < valid_len[:,None]
    X[~mask] = value
    return X

def softmax_mask(X,valid_len):
    if valid_len is None:
        return F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_len.dim()==1:
            valid_len = torch.repeat_interleave(valid_len,shape[1])
        else:
            valid_len = valid_len.reshape(-1)
        X = sequence_mask(X.reshape(-1,shape[-1]),valid_len,-1e4)
        return F.softmax(X.reshape(shape),dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout,**kwargs):
        super(DotProductAttention,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,valid_len=None):
        d = query.shape[-1]
        scores = torch.matmul(query,torch.permute(key,(0,2,1))) / math.sqrt(d)
        self.attention_weight = softmax_mask(scores,valid_len) # shape [batch_size,query_len,key-value_len]
        return torch.matmul(self.dropout(self.attention_weight),value)
    

if __name__=="__main__":
    a = torch.rand(size=(2,3,4))
    print(a)
    num_steps = 3
    batch_size=2
    mask = torch.arange(1,num_steps+1).repeat(batch_size,1)
    X = softmax_mask(a,mask)
    print(X.shape)