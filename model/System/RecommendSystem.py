import torch
from model.concat.WideAndDeep import WideAndDeep

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




# def build_recommend_dict(device):
#     print("正在生成用户推荐表")
#     # 数据需要至少六次有效观看，数据格式为
#     # [[1,2,3,4,5,6,1],
#     #  [2,3,4,5,6,7,1]]
#     WideAndDeep_net = torch.load(get_path.WideAndDeep_net_path,map_location=torch.device(device),weights_only=False)
#     WideAndDeep_net.to(device)
#     WideAndDeep_net.fmcross.device=device
#     user_seq = json.load(open(get_path.uer_sequence_path,"r+",encoding="utf-8"))
#     system = RecommendSystem(WideAndDeep_net,device)
#     recommend_dict = defaultdict(set)
    
#     if len(user_seq)==0:
#         print("没有用户互动序列")
#     else:
#         print("正在生成推荐列表")
#         for u in tqdm(user_seq):
#             torch.cuda.empty_cache()
#             seqs = np.array(user_seq[u])
#             if len(seqs) == 0 or len(seqs[seqs[:,-1]==1])==0:
#                 continue
#             valid_seqs = seqs[seqs[:,-1]==1]
#             history_seq = valid_seqs[:,:-2]
#             decoder_begin_item = valid_seqs[:,-2]
#             target_seqs = valid_seqs[:,1:-1]
#             recommend_item = system.predict(eval(u),history_seq,decoder_begin_item,target_seqs,3)
#             if len(recommend_item) ==0:
#                 continue
#             recommend_dict[u] |= recommend_item
#         pickle.dump(recommend_dict,open(get_path.recommend_dict_path,"wb"))