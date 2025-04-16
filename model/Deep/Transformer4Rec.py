from model.Deep.TransformerDecoder import TransformerDecoder
from model.Deep.TransformerEncoder import TransformerEncoder
from torch import nn
import torch

class Transformer4Rec(nn.Module):
    def __init__(self,n_item,query_size,key_size,value_size,num_hidden,
                fnn_input,fnn_hidden,all_seqs_len,
                num_heads,num_layer,dropout,**kwargs):
        super().__init__(**kwargs)
        self.item_emb = nn.Embedding(n_item,num_hidden)
        self.encoder = TransformerEncoder(query_size,key_size,value_size,num_hidden,num_heads,num_layer,fnn_input,fnn_hidden,dropout)
        self.mlp = self.__MLP4Rec(num_hidden*all_seqs_len)
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.decoder = TransformerDecoder(query_size,key_size,value_size,num_hidden,fnn_input,fnn_hidden,num_heads,dropout,num_layer)
        self.dense4seq = self.__MLP4Seq(num_hidden,n_item)
        self.crossEntoryLoss = nn.CrossEntropyLoss()
        self.n_item = n_item


    def __MLP4Rec(self,dim):
        return nn.Sequential(
            nn.Linear(dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def __MLP4Seq(self,dim,n_item):
        return nn.Sequential(
            nn.Linear(dim,n_item),
            nn.Softmax(dim=-1)
        )
    def __RecLogit(self,enc_seqs,target_item):
        target_item_emb = torch.unsqueeze(self.item_emb(target_item),dim=1)
        if target_item_emb.dim()==4:
            target_item_emb = torch.squeeze(target_item_emb,dim=1)
        all_item = torch.cat([enc_seqs,target_item_emb],dim=1)
        all_item = torch.flatten(all_item,start_dim=1)
        return torch.squeeze(self.mlp(all_item),dim=1)

    def forward4Rec(self,enc_seqs,target_item):
        return self.__RecLogit(enc_seqs,target_item)
    
    def forward4Seq(self,enc_seqs,target_seqs):
        target_seqs_emb = self.item_emb(target_seqs)
        state = self.decoder.init_state(enc_seqs)
        out,state = self.decoder(target_seqs_emb,state)
        return self.dense4seq(out),state


    def forward(self,history_sequnence,target_sequence,target_item):
        enc_seqs = self.encoder(self.item_emb(history_sequnence))
        Rec_logit = self.forward4Rec(enc_seqs,target_item)
        Seq_out,_ = self.forward4Seq(enc_seqs,target_sequence)
        return Rec_logit,Seq_out
    
    def compute_loss(self,logit,target_label,out,target_seq):
        RecLoss = self.BCEloss(logit,target_label)
        SeqLoss = self.crossEntoryLoss(out.reshape(-1,self.n_item),target_seq.reshape(-1))
        return RecLoss + SeqLoss



