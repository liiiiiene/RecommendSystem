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
from model.LLM_Rec.LLM_few_shot import LLM_FewShot
import concurrent.futures
# import multiprocessing
# import os
# import time
# from itertools import islice

prompt_prefix = """你是一个视频推荐系统。根据用户观看历史，预测用户点击新视频的概率。
"""
prompt_suffix = """
历史序列: {sequeence}
候选视频: {candidate}

请根据上述历史序列，预测用户点击候选视频的概率。
返回一个0到1之间的浮点数，只返回数字，不要有任何解释。
"""

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
            torch.cuda.empty_cache()
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


def get_response(llm,sequeence,candidate):
    for _ in range(10):
        response = llm.FewShot(sequeence,candidate)
        try:
            response = eval(response.content)
            break
        except:
            response = None
            continue
    return response

def process_candidate(args):
    llm, sequence, v, candidate_dict = args
    if str(v) not in candidate_dict.keys():
        return None
    single_candidate = ''
    single_candidate += candidate_dict[str(v)] + "\n"
    response = get_response(llm, sequence, single_candidate)
    if response is not None:
        return (v, response)
    return None

def get_llm_sequence_recommend():
    llm = LLM_FewShot(prompt_prefix,prompt_suffix,["sequeence","candidate"])
    prompt_user_dict = get_video_prompt()
    recommend_dict = pickle.load(open(get_path.recommend_dict_path,"rb"))
    candidate_dict = json.load(open(get_path.candidate_video_describe_path))
    llm_recommend_dict = defaultdict(list)
    
    # 设置线程池最大工作线程数
    max_workers = 10  # 可以根据系统性能调整
    
    for u in tqdm(recommend_dict):
        if u not in prompt_user_dict.keys():
            continue
        sequence = prompt_user_dict[u]
        candidate_video = recommend_dict[u]
        output = []
        
        # 准备参数列表
        tasks = [(llm, sequence, v, candidate_dict) for v in candidate_video]
        
        # 使用多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_candidate, tasks))
            
        # 过滤掉None结果并排序
        output = [result for result in results if result is not None]
        output = [i[0] for i in sorted(output, key=lambda x:x[1], reverse=True)]
        output = output[:5] if len(output)>=5 else output
        llm_recommend_dict[u].append(output)
    
    json.dump(llm_recommend_dict, open(get_path.llm_recommend_dict_path, "w+", encoding="utf-8"), indent=4)


if __name__=="__main__":
    # 检测所有可用的CUDA设备
    available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"可用设备: {available_devices}")
    
    # 使用所有可用设备
    build_recommend_dict(available_devices)
    get_llm_sequence_recommend()







# def build_recommend_dict_worker(worker_id, device, user_subset, result_queue):
#     print(f"进程 {worker_id} 使用 {device} 处理 {len(user_subset)} 个用户")
#     WideAndDeep_net = torch.load(get_path.WideAndDeep_net_path, map_location=torch.device(device), weights_only=False)
#     WideAndDeep_net.to(device)
#     WideAndDeep_net.fmcross.device = device
#     system = RecommendSystem(WideAndDeep_net, device)
#     recommend_dict = defaultdict(set)
    
#     for u in tqdm(user_subset, desc=f"进程 {worker_id} 在 {device}"):
#         torch.cuda.empty_cache()
#         seqs = np.array(user_subset[u])
#         if len(seqs) == 0 or len(seqs[seqs[:,-1]==1])==0:
#             continue
#         valid_seqs = seqs[seqs[:,-1]==1]
#         history_seq = valid_seqs[:,:-2]
#         decoder_begin_item = valid_seqs[:,-2]
#         target_seqs = valid_seqs[:,1:-1]
#         recommend_item = system.predict(eval(u), history_seq, decoder_begin_item, target_seqs, 3)
#         if len(recommend_item) == 0:
#             continue
#         recommend_dict[u] |= recommend_item
    
#     result_queue.put(recommend_dict)

# def split_dict(data, n):
#     """将字典分成n个子字典"""
#     result = []
#     items_per_dict = max(1, len(data) // n)
#     dict_items = list(data.items())
    
#     for i in range(0, len(dict_items), items_per_dict):
#         chunk = dict(dict_items[i:i + items_per_dict])
#         if chunk:  # 确保不添加空字典
#             result.append(chunk)
    
#     return result[:n]  # 确保最多返回n个子字典

# def build_recommend_dict(devices=None):
#     print("正在生成用户推荐表")
#     # 数据需要至少六次有效观看，数据格式为
#     # [[1,2,3,4,5,6,1],
#     #  [2,3,4,5,6,7,1]]
#     # 前五次有效观看为 transformer encoder 部分，
#     # 最后一次有效观看是 transformer decoder 是 decoder_x
    
#     # 若未指定设备，则使用所有可用的CUDA设备
#     if devices is None:
#         devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
#     if len(devices) == 0:
#         devices = ["cpu"]
#         print("未检测到CUDA设备，使用CPU")
    
#     print(f"使用设备: {devices}")
    
#     user_seq = json.load(open(get_path.uer_sequence_path, "r+", encoding="utf-8"))
    
#     if len(user_seq) == 0:
#         print("没有用户互动序列")
#         return
    
#     # 将用户序列分成与设备数量相同的批次
#     user_batches = split_dict(user_seq, len(devices))
    
#     # 创建进程间通信的队列
#     result_queue = multiprocessing.Queue()
    
#     # 创建并启动进程
#     processes = []
#     for i, (device, user_batch) in enumerate(zip(devices, user_batches)):
#         p = multiprocessing.Process(
#             target=build_recommend_dict_worker,
#             args=(i, device, user_batch, result_queue)
#         )
#         processes.append(p)
#         p.start()
    
#     # 收集所有进程的结果
#     recommend_dict = defaultdict(set)
#     for _ in range(len(processes)):
#         worker_result = result_queue.get()
#         for user, items in worker_result.items():
#             recommend_dict[user] |= items
    
#     # 等待所有进程完成
#     for p in processes:
#         p.join()
    
#     # 保存结果
#     pickle.dump(recommend_dict, open(get_path.recommend_dict_path, "wb"))
#     print(f"推荐表生成完成，共 {len(recommend_dict)} 个用户")