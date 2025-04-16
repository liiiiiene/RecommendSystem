from torch import nn


class FeedForward(nn.Module):
    def __init__(self,fnn_input,fnn_hidden,fnn_output,dropout,**kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(fnn_input,fnn_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(fnn_hidden,fnn_output)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.dense2(self.dropout(self.relu(self.dense1(x))))