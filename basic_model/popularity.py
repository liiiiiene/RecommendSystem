import pandas as pd
import matplotlib.pyplot as plt
import collections

def get_popularity(k):
    df = pd.read_csv(r"data\small_matrix.csv",encoding="utf-8")
    df = df[df["play_duration"].apply(lambda x:isinstance(x,int))&
            df["play_duration"]!=0]
    ppl_counter = df.groupby("video_id")["watch_ratio"].sum().to_dict()
    popularity = [j[0] for j in sorted([i for i in ppl_counter.items()],key=lambda x:x[1],reverse=True)[:k]]
    views_counter = collections.Counter(df["video_id"])
    popularity_list = []
    for id in popularity:
        views = views_counter[id]
        popularity_list.append((id,views))
    return popularity_list

def get_title():
    title_df = pd.read_csv(r"data\kuairec_caption_category.csv", engine='python', sep=",")
    id_caption = title_df.set_index("video_id")["caption"].to_dict()
    return id_caption

def get_popularity_video(popularity,id_caption):
    title = []
    total_views = []
    for id,views in popularity:
        t = id_caption[str(id)]
        # if not isinstance(t,float):
        #     title.append(t)
        # else:
        #     title.append(str(id))
        title.append(str(id))
        total_views.append(views)
    # 处理中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    plt.bar(range(len(title)),total_views,align="center")
    plt.xticks(range(len(title)),title, rotation=45, ha='right')
    plt.show()

if __name__=="__main__":
    popularity_list = get_popularity(10)
    title = get_title()
    get_popularity_video(popularity_list,title)