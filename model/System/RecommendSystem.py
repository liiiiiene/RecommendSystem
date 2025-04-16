import torch
from torch import nn
from load_data import get_path
from model.concat.WideAndDeep import WideAndDeep
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from load_data.LLM_utils import get_video_prompt
import pickle

class RecommendSystem:
    def __init__(self,net:WideAndDeep,device,alpha=0.5):
        self.net = net
        self.alpha = alpha
        self.device = device
        self.net.eval()

    def rate_mask(self,logits,seqs,value=0.5):
        mask = logits > value
        return seqs[mask]

    def predict(self,uid,history_seqs,decoder_begin_seqs,target_seqs,num_predict):
        target_seqs = torch.LongTensor(target_seqs).to(self.device)
        history_seqs = torch.LongTensor(history_seqs).to(self.device)
        decoder_begin_seqs = torch.LongTensor(decoder_begin_seqs).unsqueeze(dim=1).to(self.device)
        encoder_seqs = self.net.transformer.encoder(self.net.transformer.item_emb(history_seqs))
        decoder_out,decoder_state = self.net.transformer.forward4Seq(encoder_seqs,target_seqs)
        decoder_out = torch.argmax(decoder_out,dim=2).type(torch.int64)[:,-1]
        output = []
        output.append(decoder_out.reshape(-1,1))
        
        uid = [uid]*encoder_seqs.shape[0]
        for i in range(num_predict):
            y,decoder_state = self.net.transformer.decoder(self.net.transformer.item_emb(decoder_out).unsqueeze(dim=1),decoder_state)
            y = self.net.transformer.dense4seq(y)
            decoder_out = torch.argmax(y,dim=2).type(torch.int64).reshape(-1)
            output.append(decoder_out.reshape(-1,1))
        
        pre_seqs = torch.cat(output,dim=1)

        # history_set = set(j.item() for i in history_seqs for j in i) | set(j.item() for i in target_seqs for j in i)
        recommend_item = set()
        for i in range(pre_seqs.shape[1]-1):
            transformer_ctr = self.net.transformer.forward4Rec(encoder_seqs[:,i+1:],pre_seqs[:,:i+2])
            Fm_ctr = self.net.fmcross(uid,pre_seqs[:,i+1])
            ctr_out = self.net.Logit(self.net.project((Fm_ctr + transformer_ctr).reshape(-1,1)).unsqueeze(dim=1)).squeeze(dim=1)
            _,top5_indices = torch.topk(ctr_out,k=5)
            recommend_item |= set(i.item() for i in pre_seqs[:,i+1][top5_indices])
        return recommend_item


def build_recommend_dict(device):
    print("正在生成用户推荐表")
    # 数据需要至少六次有效观看，数据格式为
    # [[1,2,3,4,5,6,1],
    #  [2,3,4,5,6,7,1]]
    # 前五次有效观看为 transformer encoder 部分，
    # 最后一次有效观看是 transformer decoder 是 decoder_x

    WideAndDeep_net = torch.load(get_path.WideAndDeep_net_path,map_location=torch.device(device),weights_only=False)
    WideAndDeep_net.to(device)
    WideAndDeep_net.fmcross.device=device
    user_seq = json.load(open(get_path.uer_sequence_path,"r+",encoding="utf-8"))
    system = RecommendSystem(WideAndDeep_net,device)
    recommend_dict = defaultdict(set)
    
    if len(user_seq)==0:
        print("没有用户互动序列")
    else:
        print("正在生成推荐列表")
        for u in tqdm(user_seq):
            seqs = np.array(user_seq[u])
            if len(seqs) == 0 or len(seqs[seqs[:,-1]==1])==0:
                continue
            valid_seqs = seqs[seqs[:,-1]==1]
            history_seq = valid_seqs[:,:-2]
            decoder_begin_item = valid_seqs[:,-2]
            target_seqs = valid_seqs[:,1:-1]
            recommend_item = system.predict(eval(u),history_seq,decoder_begin_item,target_seqs,3)
            if len(recommend_item) ==0:
                continue
            recommend_dict[u] |= recommend_item
        pickle.dump(recommend_dict,open(get_path.recommend_dict_path,"wb"))


def get_llm_sequence_recommend():
    prompt_user_dict = get_video_prompt()
    recommend_dict = pickle.load(open(get_path.recommend_dict_path,"rb"))
    candidate_dict = json.load(open(get_path.candidate_video_describe_path))
    for u in tqdm(recommend_dict):
        prompt = prompt_user_dict[u]
        prompt += "以下是候选的推荐视频: \n"
        candidate_video = recommend_dict[u]
        prompt += "以下是候选物品的视频信息：\n"
        for v in candidate_video:
            prompt += candidate_dict[v] + "\n"
    prompt += """请为我根据观看的序列，对候选的推荐物品进行推荐排序，并返回一个最大长度为5的列表（如果候选的推荐视频
    不足5个，仅作排序，不做筛选），列表的第一个是你认为最适合推荐的，随后推荐的适合度依次递减，仅返回推荐适合度最高的5个物品"""

if __name__=="__main__":
    device = "cuda:0"
    build_recommend_dict(device)
    
