import load_data.get_path as get_path
import sqlite3
import os
import pandas as pd
import torch
from model.Wide.FeatureCross import FeatureCross



class ColdStartSystem:
    def __init__(self,net:FeatureCross,user_name,device) -> None:
        self.net = net
        self.net.device = device

        db_name = os.path.join(get_path.sqlite_folder_path,f"{user_name}.sqlite")
        conn = sqlite3.connect(db_name)
        user_id = pd.read_sql("""SELECT index_id FROM UserInfo""", conn).loc[0].values
        conn.close()
        self.item_id = pd.read_csv(get_path.item_feature_path,index_col=0).index
        self.user_id = user_id.repeat(len(self.item_id))
        
    def get_populary(self):
        df = pd.read_csv(r"data\small_matrix.csv",encoding="utf-8")
        df = df[df["play_duration"].apply(lambda x:isinstance(x,int))&
        df["play_duration"]!=0]
        ppl_counter = df.groupby("video_id")["watch_ratio"].sum().to_dict()
        popularity_list = [j[0] for j in sorted([i for i in ppl_counter.items()],key=lambda x:x[1],reverse=True)]
        return popularity_list

    def predict(self):
        ret = self.net(self.user_id,self.item_id)
        value = 0.5
        indices = torch.where(ret>value)[0].to("cpu").tolist()
        if(len(indices)<100):
            length = 100 - len(indices)
            list_pop = self.get_populary()
            popularity_list = [x for x in list_pop if x not in indices][:length]
            indices.extend(popularity_list)
        return indices
    
    def recommend_dict(self,indices):
        df = pd.read_csv(get_path.video_title_path)
        item_title = df.loc[indices]["text"].values
        rec_dict = {}
        for i in range(len(indices)):
            rec_dict[indices[i]] = item_title[i]
        return rec_dict
        

if __name__=="__main__":
    device = 'cuda:0'
    net = torch.load(get_path.FMcross_net_path, map_location=torch.device(device), weights_only=False)
    cs = ColdStartSystem(net,"libene",device)
    indices = cs.predict()
    rec_dict = cs.recommend_dict(indices)



