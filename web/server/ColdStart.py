from model.cold_start.cold_start_model import ColdStartSystem
import torch

def cold_start(username):
    device = 'cuda:0'
    net = torch.load("E:/Graduation_Project/MyRecommendSystem/process_data/FM_cross_net.pt", map_location=torch.device(device), weights_only=False)
    cs = ColdStartSystem(net,username,device)
    indices = cs.predict()
    rec_dict = cs.recommend_dict(indices)
    return rec_dict
