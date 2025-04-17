from model.Deep.Transformer4Rec import Transformer4Rec
from model.Wide.FeatureCross import FeatureCross
from torch import nn
from model.Deep.Transformer4Rec import Transformer4Rec


class WideAndDeep(nn.Module):

    def __init__(self,n_item,query_size,key_size,value_size,num_hidden,
                fnn_input,fnn_hidden,all_seqs_len,num_heads,num_layer,dropout,
                user_feature, item_feature, gpt_embedding, 
                n_user_feature, n_item_feature, device,**kwargs):
        super().__init__(**kwargs)
        self.transformer = Transformer4Rec(n_item,query_size,key_size,value_size,
                                        num_hidden,fnn_input,fnn_hidden,all_seqs_len,
                                        num_heads,num_layer,dropout)
        
        self.fmcross = FeatureCross(user_feature,item_feature,gpt_embedding,
                                    n_user_feature,n_item_feature,num_hidden,device)
        
        # 两层残差
        self.resnet1 = self.__residual(1,num_hidden,num_hidden)
        self.reshapeX1 = nn.Conv1d(1,num_hidden,kernel_size=1)
        self.resnet2 = self.__residual(num_hidden,num_hidden*2,num_hidden)
        self.reshapeX2 = nn.Conv1d(num_hidden,num_hidden*2,kernel_size=1)
        self.mlp = self.__mlp(num_hidden*2)

        self.project = nn.Linear(1,num_hidden)

    def Logit(self,X):
        y = self.resnet1(X)
        y += nn.functional.relu(self.reshapeX1(X))
        y1 = self.resnet2(y)
        y1 += nn.functional.relu(self.reshapeX2(y))
        y1 = self.mlp(y1)
        return y1
    
    def __residual(self,input_dim,output_dim,normal_shape):
        return nn.Sequential(
            nn.Conv1d(input_dim,output_dim,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.LayerNorm([output_dim,normal_shape]),
            nn.Conv1d(output_dim,output_dim,kernel_size=3,padding=1,stride=1),
            nn.LayerNorm([output_dim,normal_shape])
        )
    
    def __mlp(self,input):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input,1),
            nn.Sigmoid()
        )

    def forward(self,uid,history_seq,target_seq,target_item):
        fm_logits = self.fmcross(uid,target_item)
        tran_logits,tran_seq = self.transformer(history_seq,target_seq,target_item)
        logits = self.Logit(self.project((fm_logits + tran_logits).reshape(-1,1)).unsqueeze(dim=1))
        return logits.squeeze(dim=1),tran_seq
    
    def loss(self,logits,target_label,seqs_out,target_seq):
        return self.transformer.compute_loss(logits,target_label,seqs_out,target_seq)


