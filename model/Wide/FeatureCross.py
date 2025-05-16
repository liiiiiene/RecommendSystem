import torch
from torch import nn


# wide 部分交叉项
class FeatureCross(nn.Module):
    def __init__(self, user_feature, item_feature, gpt_embedding, 
                n_user_feature, n_item_feature, num_hiddens, 
                device, **kwargs):
        super(FeatureCross, self).__init__(**kwargs)
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.projection = nn.Linear(num_hiddens+gpt_embedding.shape[1],num_hiddens)
        self.user_embedding = nn.Embedding(n_user_feature, num_hiddens)
        self.item_embedding = nn.Embedding(n_item_feature, num_hiddens)
        self.device = device
        self.gpt_embedding_layer = nn.Embedding.from_pretrained(
            gpt_embedding,
            freeze=False  # True=保持固定，False=允许微调
        )
        self.dense = self.__MLP(num_hiddens*(self.user_feature.shape[1] + self.item_feature.shape[1]))
        self.project_out = nn.Sequential(nn.Linear(2,1),nn.Sigmoid())

    def __concat_feature(self, u, i):
        if isinstance(i,torch.Tensor) and i.device!="cpu":
            i = i.to("cpu")
        if isinstance(u,torch.Tensor) and u.device!="cpu":
            u = u.to("cpu")
        uid_feature = torch.LongTensor(self.user_feature.loc[u].values).to(self.device)
        iid_feature = torch.LongTensor(self.item_feature.loc[i].values).to(self.device)
        
        u_mask = (uid_feature != -1).type(torch.int)
        i_mask = (iid_feature != -1).type(torch.int)
        uid_emb = self.user_embedding(uid_feature * u_mask) * torch.unsqueeze(u_mask, dim=2) * 0.5
        iid_emb = self.item_embedding(iid_feature * i_mask) * torch.unsqueeze(i_mask, dim=2) 
        gpt_emb = self.gpt_embedding_layer(iid_feature * i_mask) * torch.unsqueeze(i_mask, dim=2) 
        iid_emb = self.projection(torch.cat((iid_emb,gpt_emb),dim=2).type(torch.float32)) * 0.5

        all_feature = torch.cat((uid_emb, iid_emb), dim=1)

        return all_feature
    
    def __MLP(self,dim):
        return nn.Sequential(
            nn.Linear(dim,dim//2),
            nn.ReLU(),
            nn.Linear(dim//2,dim//4),
            nn.ReLU(),
            nn.Linear(dim//4,1),
            nn.Sigmoid()
        )

    def forward(self,uid,iid):
        mask_emb = self.__concat_feature(uid,iid)

        # FM交叉项
        square_of_sum = torch.sum(mask_emb,dim=1)**2
        sum_of_square = torch.sum(mask_emb**2,dim=1)
        output = (torch.sum(square_of_sum - sum_of_square,dim=1)*0.5).reshape(-1,1)
        out = self.dense(mask_emb.reshape(output.shape[0],-1)).reshape(-1,1)
        result = self.project_out(torch.cat([out,output],dim=1))
        return result.reshape(-1)