import load_data.get_path as get_path
import json



def init_Best(predict_path,parameter_path,label):
    predict = json.load(open(predict_path,"r+",encoding="utf-8"))
    predict[label]["precision"] = 0
    predict[label]["recall"] = 0
    predict[label]["accuracy"] = 0
    predict[label]["loss"] = 1e4
    json.dump(predict,open(predict_path,"w+",encoding="utf-8"),indent=4)

    parameter = json.load(open(parameter_path,"r+",encoding="utf-8"))
    parameter[label]["num_epochs"] = 0
    parameter[label]["lr"] = 0
    parameter[label]["batch_size"] = 0
    parameter[label]["weight_decay"] = 0
    parameter[label]["num_hidden"] = 0
    parameter[label]["patience"] = 0
    try:
        parameter[label]["num_head"] = 0
        parameter[label]["num_layer"] = 0
        parameter[label]["dropout"] = 0
    except:
        pass
    
    json.dump(parameter,open(parameter_path,"w+",encoding="utf-8"),indent=4)

    return predict,parameter

if __name__=="__main__":
    Wide = (get_path.FM_BsetPredict_path,get_path.FM_BestParameter_path,"FMcross")
    Deep = (get_path.Transformer_BestPredict_path,get_path.Transformer_BestParameter_path,"transformer")
    WideAndDeep = (get_path.WideAndDeep_BestPredict_path,get_path.WideAndDeep_BestParameter_path,"WideAndDeep")
    item = [Wide,Deep,WideAndDeep]
    for i in item:
        init_Best(*i)


