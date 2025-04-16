from torch import nn


class AddNorm(nn.Module):
    def __init__(self,num_hidden,dropout,**kwargs):
        super().__init__(**kwargs)
        self.ln = nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self,X,Y):
        return self.ln(self.dropout(Y) + X)