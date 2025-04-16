import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1+np.exp(-x))


def compute_sim(s1,s2,ppl,index):
    alpha = (1+ppl[index])/2
    return len(s1&s2) / (len(s1)**(1-alpha) * len(s2)**(alpha))


def precison4set(real_pos,real_neg,pred_pos):
    TP = len(real_pos&pred_pos)
    FP = len(pred_pos&real_neg)
    return TP / (TP+FP) if (TP+FP)!=0 else None


def recall4set(real_pos,pred_pos):
    return len(real_pos&pred_pos) / len(real_pos)


def get_data(test_ratio=0.1):
    df = pd.read_csv(r"process_data\triples.csv",encoding="utf-8")
    test_df = df.sample(frac=test_ratio,random_state=40)
    train_df = df[~df.index.isin(test_df.index)]

    user_item = train_df.groupby("user_id")["video_id"].apply(set).to_dict()
    ppl_df = train_df.groupby("user_id")["watch_ratio"].sum()
    ppl_user = sigmoid(np.log1p(ppl_df)).to_dict()

    item_user = train_df.groupby("video_id")["user_id"].apply(set).to_dict()
    ppl_df = train_df.groupby("video_id")["watch_ratio"].sum()
    ppl_item = sigmoid(np.log1p(ppl_df)).to_dict()
    return ((user_item,ppl_user),(item_user,ppl_item)),test_df


def evaluate(test_df,recommand_dict):
    all_user_item = defaultdict(set)
    real_pos = defaultdict(set)
    real_neg = defaultdict(set)

    for user_id,group in test_df.groupby("user_id"):
        all_user_item[user_id] = set(group["video_id"])
        real_pos[user_id] = set(group[group["watch_ratio"]>=0.5]["video_id"])
        real_neg[user_id] = set(group[group["watch_ratio"]<0.5]["video_id"])

    pre = 0
    rec = 0
    total = 0
    qbar = tqdm(total=len(real_pos),desc="正在进行评估")
    for u in real_pos:
        if u not in recommand_dict or len(recommand_dict[u])==0:
            continue
        p = precison4set(real_pos[u],real_neg[u],recommand_dict[u])
        if p:
            pre +=p
        rec += recall4set(real_pos[u],recommand_dict[u])
        total +=1
        qbar.update(1)
    print(f"precison:{pre / total}\nrecall:{rec / total}")