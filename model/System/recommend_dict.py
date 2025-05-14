import torch
from load_data import get_path
from model.concat.WideAndDeep import WideAndDeep
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import torch.multiprocessing as mp
from model.System.RecommendSystem import RecommendSystem

def build_recommend_dict_worker(worker_id, device, user_subset, result_queue):
    print(f"进程 {worker_id} 使用 {device} 处理 {len(user_subset)} 个用户")
    WideAndDeep_net = torch.load(get_path.WideAndDeep_net_path, map_location=torch.device(device), weights_only=False)
    WideAndDeep_net.to(device)
    WideAndDeep_net.fmcross.device = device
    system = RecommendSystem(WideAndDeep_net, device)
    recommend_dict = defaultdict(set)
    
    for u in tqdm(user_subset, desc=f"进程 {worker_id} 在 {device}"):
        torch.cuda.empty_cache()
        seqs = np.array(user_subset[u])
        if len(seqs) == 0 or len(seqs[seqs[:,-1]==1])==0:
            continue
        valid_seqs = seqs[seqs[:,-1]==1]
        history_seq = valid_seqs[:,:-2]
        target_seqs = valid_seqs[:,1:-1]
        recommend_item = system.predict(eval(u), history_seq, target_seqs, 3)
        if len(recommend_item) == 0:
            continue
        recommend_dict[u] |= recommend_item
    
    result_queue.put(recommend_dict)


def split_user_seqs(user_seqs,n):
    
    slice_step = max(1,len(user_seqs)//n)
    total_list = list(user_seqs.items())
    result = [dict() for _ in range(n)]

    counter = 0
    for i in range(0,len(total_list),slice_step):
        item_dict = dict(total_list[i:min(i+slice_step,len(total_list))])
        result[counter%n].update(item_dict)
        counter += 1
    return result


def build_recommend_dict(avaliable_device=None,open_path = get_path.user_sequence_path,save_path = get_path.recommend_dict_path):
    mp.set_start_method('spawn', force=True)
    if avaliable_device is None:
        avaliable_device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    num_gpu = len(avaliable_device)

    user_seqs = json.load(open(open_path))

    if len(user_seqs)==0:
        print("没有用户序列")
        return
    
    gpu_tasks = split_user_seqs(user_seqs,num_gpu)

    result_quene = mp.Queue()
    

    process = []
    for i,(device,task) in enumerate(zip(avaliable_device,gpu_tasks)):
        p = mp.Process(
            target=build_recommend_dict_worker,
            args=(i,device,task,result_quene)
        )
        process.append(p)
        p.start()
    
    for p in process:
        p.join()
    
    recommend_dict = defaultdict(set)
    for _ in range(len(process)):
        result_woker = result_quene.get()
        for user,items in result_woker.items():
            recommend_dict[user] |= items
    
    # 保存结果
    pickle.dump(recommend_dict, open(save_path, "wb"))
    print(f"推荐表生成完成，共 {len(recommend_dict)} 个用户")

if __name__=="__main__":
    build_recommend_dict()