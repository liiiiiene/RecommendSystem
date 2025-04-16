import pandas as pd
import json
import load_data.get_path as get_path
import os
import shutil


def save_Best(net_path,predict_path,parameter_path,label):
    predict = json.load(open(predict_path,"r+",encoding="utf-8"))[label]
    precision = predict["precision"]
    parameter = json.load(open(parameter_path,"r+",encoding="utf-8"))[label]
    total_data = {"label":label}
    total_data.update(predict)
    total_data.update(parameter)
    (mode,head) = ("a",False) if os.path.exists(get_path.SaveBest_path) else ("w",True)
    pd.DataFrame([total_data]).to_csv(get_path.SaveBest_path,sep=",",mode=mode,header=head,index=False)

    folder_path = f"Best_net/{label}"
    os.makedirs(folder_path,exist_ok=True)
    net_name = f"precision={precision}.pt"
    target_net_path = os.path.join(folder_path,net_name)
    shutil.copy(net_path,target_net_path)


if __name__=="__main__":
    save_Best(get_path.WideAndDeep_net_path,get_path.WideAndDeep_BestPredict_path,get_path.WideAndDeep_BestParameter_path,"WideAndDeep")