import pandas as pd
import load_data.get_path as get_path


def process_text(x):
    final_text = ""
    for head in x.columns:
        final_text += " \n " + head + ":"
        for value in x[head].values:
            if not pd.isna(value):
                final_text += value + ","
    return final_text
    
def get_item_title():

    df = pd.read_csv(
        get_path.KUAIREC_CAPTION_CATEGORY,
        engine='python',          # 使用Python解析引擎
        encoding="utf-8"
    )
    df = df[df["video_id"].notna() & df["video_id"].str.match(r'^\d+$')]
    df["video_id"] = df["video_id"].astype(int)
    
    text_head = ["manual_cover_text","caption","topic_tag","first_level_category_name",
                "second_level_category_name","third_level_category_name"]
    title_dict = df.groupby("video_id")[text_head].apply(process_text)
    data = pd.DataFrame(title_dict,columns=["text"])
    data["embedding"] = [None]*len(data)
    data.to_csv(get_path.video_title_path)

if __name__=="__main__":
    get_item_title()
