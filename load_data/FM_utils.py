import pandas as pd
from collections import defaultdict
import load_data.get_path as get_path
import json
import torch

def save_feature(feature_dict,feature_path,feature_name):
    pd.DataFrame(feature_dict).T.to_csv(feature_path)
    total_feature = set([i for li in feature_dict.values() for i in li])
    num_feature = json.load(open(get_path.deployment,"r+",encoding="utf-8"))
    num_feature[1]["num_feature"][feature_name]= len(total_feature)
    json.dump(num_feature,open(get_path.deployment,"w+",encoding="utf-8"),indent=4)


def get_item_feature():
    df = pd.read_csv(get_path.item_df_path,index_col=0)
    df = df.fillna("[]")
    feature_dict = df.groupby("video_id")["feat"].apply(lambda x: eval(x.values[0]) if not pd.isna(x.values[0]) and isinstance(x.values[0],str) else []).to_dict()
    
    max_lenght = max(len(i)for i in feature_dict.values())

    final_item_feature = defaultdict(list)
    for key,values in feature_dict.items():
        m = [-1]*max_lenght
        for i in range(len(values)):
            m[i] = values[i]
        final_item_feature[key] = m

    save_feature(final_item_feature,get_path.item_feature_path,"item_feature")


def get_user_feature():
    df = pd.read_csv(get_path.user_comsum_path,index_col=0)
    df = df.fillna(-1)
    feature_dict = dict()
    for index,row in df.iterrows():
        values = [int(val) for val in row if not pd.isna(val)]
        feature_dict[index] = values

    save_feature(feature_dict,get_path.user_feature_path,"user_feature")


def get_save_feature():
    get_item_feature()
    get_user_feature()

def get_feature():
    item_feature = pd.read_csv(get_path.item_feature_path,index_col=0)
    user_feature = pd.read_csv(get_path.user_feature_path,index_col=0)
    num_feature = json.load(open(get_path.deployment,"r",encoding="utf-8"))[1]["num_feature"]
    num_item_feature = num_feature["item_feature"]
    num_user_feature = num_feature["user_feature"]
    return item_feature,user_feature,num_item_feature,num_user_feature

def get_data():
    train_df = pd.read_csv(get_path.train_df_path)
    test_df = pd.read_csv(get_path.test_df_path)
    return train_df.values.tolist(),test_df.values.tolist()

def tensor_embedding():
    df = pd.read_csv(get_path.gpt_embeddings_path,index_col=0)
    max_id = max(df.index) + 1
    num_feature = len(eval(df.loc[0,"embedding"]))
    pre_training_embdding = torch.zeros(size=(max_id,num_feature),dtype=torch.float32)
    for i in df.index:
        pre_training_embdding[i] += torch.tensor(eval(df.loc[i,"embedding"]))
    torch.save(pre_training_embdding,get_path.embedding_tensor)

def get_emb_tensor():
    return torch.load(get_path.embedding_tensor)

if __name__=="__main__":
    get_feature()