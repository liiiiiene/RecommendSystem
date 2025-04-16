import torch
import json
import os
from utils import BestModel,SaveModel
import shutil
from itertools import product
import load_data.get_path as get_path
import torch.multiprocessing as mp
from utils.init_Best import init_Best
from utils.save_Best import save_Best


def build_temp_path(origin_net_path,origin_predict_path,origin_parameter_path,num_gpu,label):

    origin_predict,origin_parameter = init_Best(origin_predict_path,origin_parameter_path,label)
    origin_net = torch.load(origin_net_path, weights_only=False) if os.path.exists(origin_net_path) else None
    total_file_path = []

    for i in range(num_gpu):
        temp_folder_path = os.path.join(os.path.dirname(origin_predict_path),f"gpu{i}_{label}_deploy")
        net_path = os.path.join(temp_folder_path,f"temp_net.pt")
        predict_path = os.path.join(temp_folder_path,f"temp_predict.json")
        parameter_path = os.path.join(temp_folder_path,f"temp_parameter.json")
        os.makedirs(temp_folder_path,exist_ok=True)
        if origin_net is not None:
            torch.save(origin_net,net_path)
        json.dump(origin_predict,open(predict_path,"w+",encoding="utf-8"),indent=4)
        json.dump(origin_parameter,open(parameter_path,"w+",encoding="utf-8"),indent=4)

        total_file_path.append({
                "net_path":net_path,
                "predict_path":predict_path,
                "parameter_path":parameter_path
        })
    return total_file_path

def get_most_best_model(origin_net_path,origin_predict_path,origin_parameter_path,net_path,predict_path,parameter_path,label):
    best_model = BestModel(origin_net_path,origin_predict_path,origin_parameter_path,label,None,"cuda:0")
    model = BestModel(net_path,predict_path,parameter_path,label,None,"cuda:0")
    save_model = SaveModel(origin_net_path,origin_predict_path,origin_parameter_path,label)
    if model.best_precision > best_model.best_precision:
        save_model.save_model(model)
    # 删除临时文件
    temp_path_folder = os.path.dirname(net_path)
    shutil.rmtree(temp_path_folder)


def train_on_gpu(gpu_id, task_list, file_path, train):
    for task in task_list:
        train(*task, device=f"cuda:{gpu_id}", **file_path)

def ParametersFind(label,parameter_deploy,train):
    torch.cuda.empty_cache()

    all_combinations = list(product(*parameter_deploy))

    if label == "FMcross":
        origin_path = (get_path.FMcross_net_path,get_path.FM_BsetPredict_path,get_path.FM_BestParameter_path)
    elif label == "transformer":
        origin_path = (get_path.Transformer_net_path,get_path.Transformer_BestPredict_path,get_path.Transformer_BestParameter_path)
    elif label == "WideAndDeep":
        origin_path = (get_path.WideAndDeep_net_path,get_path.WideAndDeep_BestPredict_path,get_path.WideAndDeep_BestParameter_path)
    else:
        print("传入标签出错")
        return

    if torch.cuda.device_count() == 1:
        init_Best(origin_path[1],origin_path[2],label)
        for com in all_combinations:
            path_deploy = {
                "net_path":origin_path[0],
                "predict_path":origin_path[1],
                "parameter_path":origin_path[2]
            }
            train(*com,device="cuda",**path_deploy)
        save_Best(origin_path[0],origin_path[1],origin_path[2],label)
    else:
        mp.set_start_method('spawn', force=True)
        num_gpu = 2
        task_per_gpu = [[] for _ in range(num_gpu)]
        for i,com in enumerate(all_combinations):
            task_per_gpu[i%num_gpu].append(com) # 分配不同gpu上执行的任务数
        # 获取在不同 gpu 上训练的暂时保存的路径
        total_file_path = build_temp_path(*origin_path,num_gpu=num_gpu,label=label)
        # 在不同gpu上并行训练
        processes = []
        for gpu_id in range(num_gpu):
            p = mp.Process(
                target=train_on_gpu,
                args=(gpu_id+1, task_per_gpu[gpu_id], total_file_path[gpu_id], train)
            )
            p.start()
            processes.append(p)
            
        # 等待所有进程完成
        for p in processes:
            p.join()
        for i in range(num_gpu):
            get_most_best_model(*origin_path,**total_file_path[i],label=label)
        save_Best(origin_path[0],origin_path[1],origin_path[2],label)
        

def ParametersTest(label,parameter_deploy,train):
    print("单显卡测试模式")
    all_combinations = list(product(*parameter_deploy))
    torch.cuda.empty_cache()
    
    if label == "FMcross":
        origin_path = (get_path.FMcross_net_path,get_path.FM_BsetPredict_path,get_path.FM_BestParameter_path)
    elif label == "transformer":
        origin_path = (get_path.Transformer_net_path,get_path.Transformer_BestPredict_path,get_path.Transformer_BestParameter_path)
    elif label == "WideAndDeep":
        origin_path = (get_path.WideAndDeep_net_path,get_path.WideAndDeep_BestPredict_path,get_path.WideAndDeep_BestParameter_path)
    else:
        print("传入标签出错")
        return
    init_Best(origin_path[1],origin_path[2],label)
    for com in all_combinations:
        path_deploy = {
            "net_path":origin_path[0],
            "predict_path":origin_path[1],
            "parameter_path":origin_path[2]
        }
        train(*com,device="cuda:1",**path_deploy)
    save_Best(origin_path[0],origin_path[1],origin_path[2],label)