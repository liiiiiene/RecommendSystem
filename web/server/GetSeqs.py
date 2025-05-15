import sqlite3
import pandas as pd
from load_data.seqs_uilts import build_seqs
import json
import load_data.get_path as get_path
from model.System.recommend_dict import build_recommend_dict
import os
from model.concat.WideAndDeep import WideAndDeep
from model.System.RecommendSystem import RecommendSystem

def reaction_data(username,video_id,timestamp,watch_ratio):
    db_name = os.path.join(get_path.sqlite_folder_path,f"{username}.sqlite")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    user_id = pd.read_sql("""SELECT index_id FROM UserInfo""", conn).loc[0].values.item()
    cursor.execute('''INSERT INTO InterAction(user_id, video_id, timestamp, watch_ratio) VALUES (?,?,?,?)''' ,(user_id, video_id, timestamp, watch_ratio))
    conn.commit()
    conn.close()

def get_seq(username):
    db_name = os.path.join(get_path.sqlite_folder_path,f"{username}.sqlite")
    user_seq_path = os.path.join(get_path.sqlite_user_sequence_folder_path,f"{username}.json")
    save_pickle_path = os.path.join(get_path.sqlite_user_sequence_folder_path,f"{username}.pkl")
    conn = sqlite3.connect(db_name)
    interaction = pd.read_sql("SELECT * FROM InterAction",conn)
    sequence = []
    columns = ["timestamp","video_id","watch_ratio"]
    user_squence = interaction.groupby("user_id")[columns].apply(lambda x: build_seqs(x,sequence)).to_dict()
    json.dump(user_squence,open(user_seq_path,"w+",encoding="utf-8"),indent=4)
    return user_squence

def process_seq(username):
    user_seq_path = os.path.join(get_path.sqlite_user_sequence_folder_path,f"{username}.json")
    save_pickle_path = os.path.join(get_path.sqlite_user_sequence_folder_path,f"{username}.pkl")
    build_recommend_dict(None,user_seq_path,save_pickle_path,k=10)
    

if __name__=="__main__":
    process_seq("admin")
    import pickle
    rec_dict = pickle.load(open(os.path.join(get_path.sqlite_user_sequence_folder_path,f"admin.pkl"),"rb"))
    print(rec_dict)