import pandas as pd
from load_data import get_path
import json
import re

def get_seqs(x,video_text):
    x = x[x["timestamp"].notna()]
    x = x.sort_values(by="timestamp",axis=0)
    x = x[x["watch_ratio"]==1]
    x = x.iloc[-10:] if len(x)>=10 else x
    single_prompt=''

    for _,row in x.iterrows():
        try:
            valid_text = video_text[ video_text["video_id"]== row["video_id"].item() ]
            title = valid_text["caption"].values[0]
        except:
            continue
        single_dict = f'''（"视频id":{valid_text["video_id"].values[0].item()},"封面文字":{valid_text["manual_cover_text"].values[0]},"简介标题":{title},"标题标签":{valid_text["topic_tag"].values[0]},"一级标签":{valid_text["first_level_category_name"].values[0]},"二级标签":{valid_text["second_level_category_name"].values[0]},"三级标签":{valid_text["third_level_category_name"].values[0]}）'''
        single_prompt += str(single_dict) + "->"
        
    single_prompt = single_prompt.rstrip("->")
    return single_prompt


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
    json.dump(user_seqs,open(get_path.user_interact_prompt_path,"w+",encoding="utf-8"),indent=4,ensure_ascii=False)

def add_id(x):
    prompt = "（视频id:" + str(x.index[0]) + ","
    prompt += re.sub(r"\n","",x.iloc[0]).replace("manual_cover_text:","封面文字:").replace("caption:","简介标题:").replace("topic_tag:","标题标签:").replace("first_level_category_name:","一级标签:").replace("second_level_category_name:","二级标签:").replace("third_level_category_name:","三级标签:")
    prompt = prompt.strip().rstrip(",")
    prompt += "）"
    return prompt


def build_candidate_prompt():
    data = pd.read_csv(get_path.video_title_path)
    process_data = data.groupby("video_id")["text"].apply(add_id).to_dict()
    json.dump(process_data,open(get_path.candidate_video_describe_path,"w+",encoding="utf-8"),indent=4,ensure_ascii=False)

def get_video_prompt():
    return json.load(open(get_path.user_interact_prompt_path))

def get_candidate_prompt():
    return json.load(open(get_path.candidate_video_describe_path))


if __name__=="__main__":
    build_video_prompt()
    build_candidate_prompt()
    