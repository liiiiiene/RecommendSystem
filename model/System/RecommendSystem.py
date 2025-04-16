import torch
from torch import nn
from load_data import get_path
from model.concat.WideAndDeep import WideAndDeep
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class RecommendSystem:
    def __init__(self,net:WideAndDeep,device,alpha=0.5):
        self.net = net
        self.alpha = alpha
        self.device = device
        self.net.eval()

    def rate_mask(self,logits,seqs,value=0.5):
        mask = logits > value
        return seqs[mask]

    def predict(self,uid,history_seqs,decoder_begin_seqs,num_predict):
        
        history_seqs = torch.LongTensor(history_seqs).to(self.device)
        decoder_seqs = torch.LongTensor(decoder_begin_seqs).to(self.device)
        encoder_seqs = self.net.transformer.encoder(self.net.transformer.item_emb(history_seqs))
        decoder_state = self.net.transformer.decoder.init_state(encoder_seqs)
        output = []
        
        uid = [uid]*encoder_seqs.shape[0]
        for i in range(num_predict):
            y,decoder_state = self.net.transformer.decoder(torch.unsqueeze(decoder_seqs,dim=1),decoder_state)
            decoder_seqs = torch.argmax(y,dim=2).type(torch.int64)
            output.append(decoder_seqs)
        pre_seqs = torch.cat(output,dim=1)

        history_set = set(j.item() for i in history_seqs for j in i) | set(i.item() for i in decoder_seqs)
        recommend_item = set()
        for i in range(pre_seqs.shape[1]):
            decoder_item = torch.LongTensor(decoder_begin_seqs).to(self.device).reshape(-1,1)
            transformer_ctr = self.net.transformer.forward4Rec(encoder_seqs[:,i+1:],torch.cat([decoder_item,pre_seqs[:,:i+1]],dim=1))
            Fm_ctr = self.net.fmcross(uid,pre_seqs[:,i])
            ctr_out = self.net.Logit(self.net.project((Fm_ctr + transformer_ctr).reshape(-1,1)).unsqueeze(dim=1)).squeeze(dim=1)
            recommend_item |= set(i.item() for i in self.rate_mask(ctr_out,pre_seqs[:,i])) - history_set
        return recommend_item


def build_recommend_dict(device):
    print("正在生成用户推荐表")
    # 数据需要至少六次有效观看，数据格式为
    # [[1,2,3,4,5,6,1],
    #  [2,3,4,5,6,7,1]]
    # 前五次有效观看为 transformer encoder 部分，
    # 最后一次有效观看是 transformer decoder 是 decoder_x

    WideAndDeep_net = torch.load(get_path.WideAndDeep_net_path,map_location=torch.device(device),weights_only=False)
    user_seq = json.load(open(get_path.uer_sequence_path,"r+",encoding="utf-8"))
    system = RecommendSystem(WideAndDeep_net,device)
    recommend_dict = defaultdict(set)
    if len(user_seq)==0:
        print("没有用户互动序列")
    else:
        
        for u in tqdm(user_seq):
            seqs = np.array(user_seq[u])
            if len(seqs) == 0 or len(seqs[seqs[:,-1]==1])==0:
                continue
            valid_seqs = seqs[seqs[:,-1]==1]
            history_seq = valid_seqs[:,:-2]
            decoder_begin_item = valid_seqs[:,-2]
            recommend_item = system.predict(eval(u),history_seq,decoder_begin_item,3)
            recommend_dict[u] |= recommend_item

if __name__=="__main__":
    device = "cuda:0"
    build_recommend_dict(device)
    
