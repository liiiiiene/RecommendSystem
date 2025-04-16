# 文件路径
import pandas as pd

def getOriginalData(file_name):
    print(f"正在获取数据集{file_name.rstrip('.csv')}")
    return pd.read_csv(file_name)
