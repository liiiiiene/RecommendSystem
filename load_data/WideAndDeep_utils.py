import json
from load_data import get_path
import numpy as np
def build_user_seqs():
    user_seq = json.load(open(get_path.uer_sequence_path,"r+",encoding="utf-8"))
    user_seqs_ratio = []
    for u in user_seq:
        seqs = user_seq[u]
        if len(seqs)==0:
            continue
        u = eval(u)
        u_npy = np.array([u]*len(seqs)).reshape(-1,1)
        users_seqs = np.concat([u_npy,seqs],axis=1)
        user_seqs_ratio.append(users_seqs)
    final_user_seqs_ratio = np.concat(user_seqs_ratio,axis=0)
    np.random.shuffle(final_user_seqs_ratio)
    np.save(get_path.user_seq_ratio_path,final_user_seqs_ratio)

def get_user_seqs(test_ratio=0.1):
    total_data = np.load(get_path.user_seq_ratio_path)
    n_item = json.load(open(get_path.deployment,"r+",encoding="utf-8"))[2]["seqs_utils"]["allItem"]
    test_lenght = int(len(total_data) * test_ratio)
    test_data = total_data[:test_lenght]
    train_data = total_data[test_lenght:]
    return n_item,train_data,test_data

if __name__=="__main__":
    build_user_seqs()