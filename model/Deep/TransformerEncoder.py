
from torch import nn
from model.Deep.AddNorm import AddNorm
from model.Deep.FeedForward import FeedForward
from model.Deep.MultiHeadAttention import MultiHeadAttention
from model.Deep.PositionalEmbedding import PositionalEmbedding
import math


class EncoderBlocks(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hiddens,num_heads,dropout,fnn_input,fnn_hidden,**kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(query_size,key_size,value_size,num_hiddens,num_heads,dropout)
        self.norm1 = AddNorm(num_hiddens,dropout)
        self.fnn = FeedForward(fnn_input,fnn_hidden,num_hiddens,dropout)
        self.norm2 = AddNorm(num_hiddens,dropout)
    def forward(self,x_emb,valid_len=None):
        Y = self.norm1(x_emb,self.attention(x_emb,x_emb,x_emb,valid_len))
        return self.norm2(Y,self.fnn(Y))
    
class TransformerEncoder(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hidden,num_heads,num_layers,fnn_input,fnn_hidden,dropout,**kwargs):
        super().__init__(**kwargs)
        
        self.positon_embedding = PositionalEmbedding(num_hidden,dropout)
        self.num_hidden = num_hidden
        self.blks = nn.ModuleList([EncoderBlocks(query_size,key_size,value_size,num_hidden,num_heads,dropout,fnn_input,fnn_hidden) 
                                for _ in range(num_layers)])
    def forward(self,X,valid_len=None):
        X = self.positon_embedding(X * math.sqrt(self.num_hidden))
        self.attention_weights = [None]*len(self.blks)
        for blk in self.blks:
            X = blk(X,valid_len)
        return X
    
