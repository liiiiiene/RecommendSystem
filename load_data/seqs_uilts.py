import numpy as np
import load_data.get_path as get_path
import load_data.process_utils as process_utils
import json

def build_seqs(x,seqs):
    
    x = x[x["timestamp"].notna()]
    x = x.sort_values(by="timestamp",axis=0)
    user_seqs = []
    single = []
    for _,row in x.iterrows():
        if len(single)==5:
            single.append(row["video_id"].astype(np.int64).item())
            single.append(row["watch_ratio"].astype(np.int64).item())
            seqs.append(single)
            user_seqs.append(single)
            single = single[1:5]
        if row["watch_ratio"]==1:
            single.append(row["video_id"].astype(np.int64).item())
    return user_seqs

def get_seqs():
    valid_df = process_utils.valid_small_martix()
    sequence = []
    columns = ["timestamp","video_id","watch_ratio"]
    user_squence = valid_df.groupby("user_id")[columns].apply(lambda x: build_seqs(x,sequence)).to_dict()
    json.dump(user_squence,open(get_path.uer_sequence_path,"w+",encoding="utf-8"),indent=4)
    np.save(get_path.sequence_item_ratio_path,np.array(sequence))
    items = set(valid_df["video_id"].values.tolist())
    with open(get_path.deployment,"r+",encoding="utf-8") as f:
        allItem = json.load(f)
        allItem[2]["seqs_utils"]["allItem"] = max(items)+1
    with open(get_path.deployment,"w+",encoding="utf-8") as f:
        json.dump(allItem,f,indent=4)
        

def load_seqs(test_ratio=0.1):
    data = np.load(get_path.sequence_item_ratio_path)
    n_item = json.load(open(get_path.deployment,"r+",encoding="utf-8"))[2]["seqs_utils"]["allItem"]
    np.random.shuffle(data)
    split_number = int(len(data)*test_ratio)
    train_data = data[split_number:]
    test_data = data[:split_number]
    return train_data,test_data,n_item

if __name__=="__main__":
    get_seqs()