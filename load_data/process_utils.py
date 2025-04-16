import pandas as pd
import json
import load_data.get_path as get_path

def valid_small_martix():
    df = pd.read_csv(get_path.SMALL_MATRIX,encoding="utf-8")
    valid_df = df[df["user_id"].notna()&
                df["video_id"].notna()&
                df["user_id"].apply(lambda x:isinstance(x,int))&
                df["video_id"].apply(lambda x:isinstance(x,int))
                ]
    valid_df["watch_ratio"] = (
    ((valid_df["video_duration"] < 10000) & (valid_df["watch_ratio"] > 1)) |
    ((valid_df["video_duration"] > 10000) & (valid_df["video_duration"] < 20000) & (valid_df["watch_ratio"] > 0.7)) |
    ((valid_df["video_duration"] > 20000) & (valid_df["watch_ratio"] > 0.5))).astype(int)
    # print("少量数据测试模式")
    # valid_df = valid_df.sample(frac=0.005,random_state=40)
    valid_df.to_csv(get_path.valid_matrix)
    

def get_triple():
    valid_df = pd.read_csv(get_path.valid_matrix)
    triple = pd.concat((valid_df["user_id"],valid_df["video_id"],valid_df["watch_ratio"]),axis=1,ignore_index=True)
    triple.to_csv(get_path.triple_path,index=False,header=["user_id","video_id","watch_ratio"])


def get_item_df():
    df = pd.read_csv(get_path.ITEM_CATEGORIES,index_col=0)
    df.iloc[:,-1].to_csv(get_path.item_df_path)


def get_user_df():
    df = pd.read_csv(get_path.USER_FEATURES,index_col=0)

    user_df = df.iloc[:,12:]

    feature_range = json.load(open(get_path.deployment,"r"))[0]["comsum"]
    comsum = 0
    user_comsum = pd.DataFrame(index=df.index,columns=range(len(feature_range)))
    for i in range(len(feature_range)):
        user_comsum.iloc[:,i] = df[f"onehot_feat{i}"] + comsum
        comsum += feature_range[f"onehot_feat{i}"]
    user_df.to_csv(get_path.user_df_path)
    user_comsum.to_csv(get_path.user_comsum_path)


def split_train_test(test_ratio=0.1):
    df = pd.read_csv(get_path.triple_path)

    # # test
    # valid_df = df.sample(frac=0.025,random_state=40)
    # test_df = valid_df.sample(frac=test_ratio,random_state=40)
    # train_df = valid_df[~valid_df.index.isin(test_df.index)]
    
    test_df = df.sample(frac=test_ratio,random_state=74)
    train_df = df[~df.index.isin(test_df.index)]
    test_df.to_csv(get_path.test_df_path,index=False)
    train_df.to_csv(get_path.train_df_path,index=False)
    
if __name__=="__main__":
    split_train_test()