import time
import random
import json
import torch
import os


class Accumulator:
    def __init__(self,m):
        self.data = [0.0]*m
    def add(self,*args):
        self.data = [i + float(j) for i,j in zip(self.data,args)]
        
    def __getitem__(self,idx):
        return self.data[idx]
    def reset(self):
        m = len(self.data)
        self.data = [0.0] * m
def random_time(a,b):
    time.sleep(random.uniform(a,b))

class BestModel:
    def __init__(self,model_path,predict_path,best_parameter_path,label,net_save,device):
        self.best_parameter_path = best_parameter_path
        self.label = label
        self.device = device
        self.net_save = net_save
        if not os.path.exists(model_path):
            self.best_loss = 1e4
            self.best_precision = 0
            self.model = False
            self.accuracy = 0
            self.recall = 0
            self.parameter_dict = None
            self.best_model = None            
        else:
            self.best_model = torch.load(model_path,map_location=torch.device(device),weights_only=False)
            predict = json.load(open(predict_path,"r+",encoding="utf-8"))[self.label]
            self.best_loss = predict["loss"]
            self.best_precision = predict["precision"]
            self.accuracy = predict["accuracy"]
            self.recall = predict["recall"]
            self.parameter_dict = json.load(open(self.best_parameter_path,"r+",encoding="utf-8"))
            self.model = True


    def isBest(self,loss,precision,recall,accuracy,net,parameter_dict):
        if self.best_loss - loss > 0.05 or precision - self.best_precision > 0.02:
            self.best_loss = loss
            self.best_precision = precision
            self.net_save.load_state_dict(net.state_dict())
            self.model = True
            self.recall = recall
            self.accuracy = accuracy
            self.parameter_dict = parameter_dict
            self.best_model = None
            return True
        return False


class EarlyStop:
    def __init__(self,patience):
        self.early_stop_counter = 0
        self.patience = patience
    def check(self,flag):
        if flag:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter +=1
        if self.early_stop_counter > self.patience:
            return True
        else:
            return False

class SaveModel:
    def __init__(self,net_path,predict_path,best_parameter_path,label):
        self.net_path = net_path
        self.best_predict_path = predict_path
        self.best_parameter_path = best_parameter_path
        self.label=label

    
    def save_model(self,best_model:BestModel):
        if not best_model.model:
            print("此参数没有最佳模型")
        else:
            if best_model.best_model is None:
                torch.save(best_model.net_save,self.net_path)
            else:
                torch.save(best_model.best_model,self.net_path)
            predict = json.load(open(self.best_predict_path,"r+",encoding="utf-8"))
            predict[self.label]["loss"] = best_model.best_loss
            predict[self.label]["precision"] = best_model.best_precision
            predict[self.label]["recall"] = best_model.recall
            predict[self.label]["accuracy"] = best_model.accuracy
            json.dump(predict,open(self.best_predict_path,"w+",encoding="utf-8"),indent=4)
            json.dump(best_model.parameter_dict,open(self.best_parameter_path,"w+",encoding="utf-8"),indent=4)