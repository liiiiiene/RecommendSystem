import torch
from torch import nn

from model.Deep.DotProductAttetion import DotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hiddens,num_heads,dropout,use_bias=True,**kwargs):
        super().__init__(**kwargs)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=use_bias)
        self.W_k = nn.Linear(key_size,num_hiddens,bias=use_bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=use_bias)
        self.attention = DotProductAttention(dropout)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=use_bias)
        self.num_heads = num_heads


    def transpose_qkv(self,x,num_heads):
        shape = x.shape
        x = x.reshape((shape[0],shape[1],num_heads,-1))
        x = torch.permute(x,(0,2,1,3))
        return x.reshape(-1,x.shape[2],x.shape[3])
    

    def transpose_output(self,x,num_heads):
        x = x.reshape(-1,num_heads,x.shape[1],x.shape[2])
        x = torch.permute(x,(0,2,1,3))
        return x.reshape(x.shape[0],x.shape[1],-1)
    

    def forward(self,query,key,value,valid_len=None):
        queries = self.transpose_qkv(self.W_q(query),self.num_heads)
        keys = self.transpose_qkv(self.W_k(key),self.num_heads)
        values = self.transpose_qkv(self.W_v(value),self.num_heads)
        
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len,repeats=self.num_heads,dim=0)

        output = self.attention(queries,keys,values,valid_len)
        output = self.transpose_output(output,self.num_heads)
        return self.W_o(output)
