from model.cold_start.cold_start_model import ColdStartSystem
import torch
import load_data.get_path as get_path
import json
import os

def cold_start(username):
    device = 'cuda:0'
    net = torch.load(get_path.FMcross_net_path, map_location=torch.device(device), weights_only=False)
    cs = ColdStartSystem(net,username,device)
    indices = cs.predict()
    rec_dict = cs.recommend_dict(indices)
    rec_list = {}
    rec_list[cs.user_id[0].item()] = indices
    json.dump(rec_list,open(os.path.join(get_path.cold_rec_folder_path,f"{username}.json"),"w+",encoding="utf-8"),indent=4)
    return rec_dict

if __name__=="__main__":
    cold_start("admin")