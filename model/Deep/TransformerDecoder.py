from model.Deep.AddNorm import AddNorm
from model.Deep.MultiHeadAttention import MultiHeadAttention
from model.Deep.PositionalEmbedding import PositionalEmbedding
from model.Deep.FeedForward import FeedForward
import torch
from torch import nn
import math

class DecoderBlocks(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hidden,fnn_input,fnn_hidden,num_heads,dropout,i,**kwargs):
        super().__init__(**kwargs)
        self.attention1 = MultiHeadAttention(query_size,key_size,value_size,num_hidden,num_heads,dropout)
        self.normlayer1 = AddNorm(num_hidden,dropout)
        self.attention2 = MultiHeadAttention(query_size,key_size,value_size,num_hidden,num_heads,dropout)
        self.normlayer2 = AddNorm(num_hidden,dropout)
        self.fnn = FeedForward(fnn_input,fnn_hidden,num_hidden,dropout)
        self.normlayer3 = AddNorm(num_hidden,dropout)
        self.i = i
        
    def forward(self,X,state):
        enc_output = state[0]

        if self.training:
            batch_size,seq_len,_ = X.shape
            dec_valid_len = torch.arange(1,seq_len+1,device=X.device).repeat(batch_size,1)
        else:
            dec_valid_len=None

        if state[1][self.i] is None:
            key_value = X
        else:
            key_value = torch.cat([state[1][self.i],X],dim=1)

        state[1][self.i] = key_value

        X2 = self.attention1(X,key_value,key_value,dec_valid_len)
        Y = self.normlayer1(X,X2)
        Y2 = self.attention2(Y,enc_output,enc_output)
        Z = self.normlayer2(Y,Y2)
        return self.normlayer3(Z,self.fnn(Z)),state

class TransformerDecoder(nn.Module):
    def __init__(self,query_size,key_size,value_size,num_hidden,fnn_input,fnn_hidden,num_heads,dropout,num_layer,**kwargs):
        super().__init__(**kwargs)
        self.num_layer = num_layer
        self.blks = nn.ModuleList([DecoderBlocks(query_size,key_size,value_size,num_hidden,fnn_input,fnn_hidden,num_heads,dropout,i) for i in range(num_layer)])
        self.position_emb = PositionalEmbedding(num_hidden,dropout)
        self.num_hidden = num_hidden
        

    def init_state(self,enc_output):
        return [enc_output,[None]*self.num_layer]
    
    def forward(self,X,state):
        X = self.position_emb(X * math.sqrt(self.num_hidden))
        if X.dim()==2:
            X = torch.unsqueeze(X,dim=1)
        for blk in self.blks:
            X,state = blk(X,state)
        return X,state
    
    