import pandas as pd
from load_data import get_path
import json
import re

def get_seqs(x,video_text):
    x = x[x["timestamp"].notna()]
    x = x.sort_values(by="timestamp",axis=0)
    x = x[x["watch_ratio"]==1]
    x = x.iloc[-10:] if len(x)>=10 else x
    video_id_list = []
    single_prompt='"video_id" 代表有效观看的视频id。。"title" 代表此视频的标题。以下是按照观看时间排序的，视频历史有效观看序列：\n'

    for _,row in x.iterrows():
        try:
            title = video_text[ video_text["video_id"]== row["video_id"].item() ]["caption"].values[0]
        except:
            continue
        single_prompt += f"[video_id:{int(row['video_id'].item())},title:{title}]->"
        video_id_list.append(int(row['video_id'].item()))


    single_prompt = single_prompt.rstrip("->")
    single_prompt += "\n"

    text_prompt = '关于这些历史有效观看视频的内容描述: \n'
    valid_text = video_text[video_text["video_id"].isin(video_id_list) ]
    for _,row in valid_text.iterrows():
        single_video_dict = {
            "视频id":row["video_id"],
            "封面文字":row["manual_cover_text"],
            "简介标题":row["caption"],
            "标题标签":row["topic_tag"],
            "一级标签":row["first_level_category_name"],
            "二级标签":row["second_level_category_name"],
            "三级标签":row["third_level_category_name"]
        }
        text_prompt += str(single_video_dict) + "\n"
    return single_prompt + text_prompt


def build_video_prompt():
    video_text = pd.read_csv(
        get_path.KUAIREC_CAPTION_CATEGORY,
        engine='python',          # 使用Python解析引擎
        encoding="utf-8",
    )
    video_text = video_text[video_text["video_id"].notna() & video_text["video_id"].str.match(r'^\d+$')]
    video_text["video_id"] = video_text["video_id"].astype(int)
    small_martix = pd.read_csv(get_path.valid_matrix)
    columns = ["timestamp","video_id","watch_ratio"]
    user_seqs = small_martix.groupby("user_id")[columns].apply(lambda x : get_seqs(x,video_text)).to_dict()
    json.dump(user_seqs,open(get_path.user_interact_prompt_path,"w+",encoding="utf-8"),indent=4)

def add_id(x):
    prompt = "{视频id:" + str(x.index[0]) + ","
    prompt += re.sub(r"\n","",x.iloc[0]).replace("manual_cover_text:","封面文字:").replace("caption:","简介标题:").replace("topic_tag:","标题标签:").replace("first_level_category_name:","一级标签:").replace("second_level_category_name:","二级标签:").replace("third_level_category_name:","三级标签:")
    prompt = prompt.strip().rstrip(",")
    prompt += "}"
    return prompt


def build_candidate_prompt():
    data = pd.read_csv(get_path.video_title_path)
    process_data = data.groupby("video_id")["text"].apply(add_id).to_dict()
    json.dump(process_data,open(get_path.candidate_video_describe_path,"w+",encoding="utf-8"),indent=4)

def get_video_prompt():
    return json.load(open(get_path.user_interact_prompt_path))

def get_candidate_prompt():
    return json.load(open(get_path.candidate_video_describe_path))


if __name__=="__main__":
    build_video_prompt()
    # build_candidate_prompt()
    