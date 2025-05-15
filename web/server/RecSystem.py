import load_data.get_path as get_path
import pandas as pd
import json
import pickle
import os
from model.System.llm_recommend_dict import get_llm_sequence_recommend
from model.System.explain_llm_recommend_dict import user_explain_llm_recommend_dict
from load_data.LLM_utils import get_seqs
import sqlite3


def build_video_item(username):
    video_text = pd.read_csv(
        get_path.KUAIREC_CAPTION_CATEGORY,
        engine='python',          # 使用Python解析引擎
        encoding="utf-8",
    )
    video_text = video_text[video_text["video_id"].notna() & video_text["video_id"].str.match(r'^\d+$')]
    video_text["video_id"] = video_text["video_id"].astype(int)
    db_name = os.path.join(get_path.sqlite_folder_path,f"{username}.sqlite")
    conn = sqlite3.connect(db_name)
    interaction = pd.read_sql("SELECT * FROM InterAction",conn)
    columns = ["timestamp","video_id","watch_ratio"]
    user_seqs = interaction.groupby("user_id")[columns].apply(lambda x : get_seqs(x,video_text)).to_dict()
    json.dump(user_seqs,open(os.path.join(get_path.sqlite_interaction_pro_folder,f"{username}.json"),"w+",encoding="utf-8"),indent=4,ensure_ascii=False)

def get_prompt(username):
    return json.load(open(os.path.join(get_path.sqlite_interaction_pro_folder,f"{username}.json"),"r+",encoding="utf-8"))


def get_rec_dict(username):
    open_path = os.path.join(get_path.sqlite_user_sequence_folder_path,f"{username}.pkl")
    save_path = os.path.join(get_path.Rec_llm_folder_path,f"{username}.json")
    build_video_item(username)
    llm_recommand_dict = get_llm_sequence_recommend(get_prompt(username),open_path,save_path)
    for user_id in llm_recommand_dict.keys():
        video_id_list = llm_recommand_dict[user_id]
        df = pd.read_csv(get_path.video_title_path)
        item_title = df.loc[video_id_list]["text"].values
        rec_list = {}
        for i in range(len(video_id_list)):
            rec_list[video_id_list[i]] = item_title[i]
    return rec_list

def explain_llm(username):
    open_path = os.path.join(get_path.Rec_llm_folder_path,f"{username}.json")
    save_path = os.path.join(get_path.Rec_explain_llm_folder_path,f"{username}.json")
    build_video_item(username)
    llm_recommand_dict = user_explain_llm_recommend_dict(get_prompt(username),open_path,save_path)
    for user_id in llm_recommand_dict.keys():
        rec_list = llm_recommand_dict[user_id]
    return rec_list

if __name__=="__main__":
    # get_rec_dict("admin")
    get_rec_dict("user")