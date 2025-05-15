from model.Wide.FeatureCross import FeatureCross
import torch
from torch import nn
from load_data import FM_utils
from torch.utils.data import DataLoader
from utils.utils import Accumulator
import load_data.get_path as get_path
from sklearn.metrics import precision_score,recall_score,accuracy_score
from utils.utils import BestModel,EarlyStop,SaveModel
import json
from tqdm import tqdm
from utils.multi_gpu_train import ParametersFind,ParametersTest


def doeval(test_data,net:FeatureCross,batch_size,value=0.5):
    net.eval()
    metric = Accumulator(4)
    for u,i,r in DataLoader(test_data,batch_size=batch_size,shuffle=True):
        logits = net(u,i)
        logits = (net(u,i) >= value).type(torch.int).to("cpu")
        precicion = precision_score(r,logits,zero_division=0)
        recall = recall_score(r,logits,zero_division=0)
        accuracy = accuracy_score(r,logits)

        metric.add(precicion,recall,accuracy,1)
    net.train()
    return metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3]


def train(num_epochs,lr,batch_size,weight_decay,num_hidden,patience,device,net_path,predict_path,parameter_path):


    TrainParameter = {
        "FMcross":{
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "num_hidden": num_hidden,
            "patience": patience,
            "num_head": -1,
            "num_layer":-1,
            "dropout":-1
            }
        }


    item_feature,user_feature,num_item_feature,num_user_feature = FM_utils.get_feature()
    gpt_embedding = FM_utils.get_emb_tensor()
    net = FeatureCross(user_feature,item_feature,gpt_embedding,num_user_feature,num_item_feature,num_hidden,device)
    
    net = net.to(device)
    train_data,test_data = FM_utils.get_data()
    loss = nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)

    # 储存最佳模型参数
    net_save = FeatureCross(user_feature,item_feature,gpt_embedding,num_user_feature,num_item_feature,num_hidden,device)
    best_model = BestModel(net_path,predict_path,parameter_path,"FMcross",net_save,device)

    save_model = SaveModel(net_path,predict_path,parameter_path,"FMcross")

    # 早停
    early_stop = EarlyStop(patience)
    
    qbar = tqdm(range(num_epochs),desc=f"batch_size:{batch_size},lr:{lr},weight_decay:{weight_decay},num_hidden:{num_hidden},patience:{patience}")
    for epoch in qbar:
        metric = Accumulator(2)
        for u,i,r in DataLoader(train_data,batch_size=batch_size,shuffle=True):
            r = r.to(device).type(torch.float32)
            optimizer.zero_grad()
            logits = net(u,i)
            l = loss(logits,r)
            l.backward()
            optimizer.step()
            metric.add(l.item(),1)

        # 每个epoch都进行评估，以便及时应用早停
        precision,recall,accuracy = doeval(test_data,net,batch_size)
        
        # 检查是否有改善
        
        if early_stop.check(best_model.isBest(metric[0]/metric[1],precision,recall,accuracy,net,TrainParameter)):
            print(f"早停：连续{patience}个epoch没有改善，在第{epoch}个epoch停止训练")
            break

        qbar.set_postfix({
                'best_loss': f'{best_model.best_loss:.4f}',
                'best_precision': f'{best_model.best_precision:.4f}'
            })
        
    save_model.save_model(best_model)

def get_FMparameters_deploy():
    optional_parameter = json.load(open(get_path.FM_ParametersDeploy_path))
    lrs = optional_parameter["lr"]
    batch_sizes = optional_parameter["batch_size"]
    weight_decays = optional_parameter["weight_decay"]
    num_hiddens = optional_parameter["num_hidden"]
    patiences = optional_parameter["patience"]
    num_epochs = optional_parameter["num_epochs"]
    return num_epochs,lrs,batch_sizes,weight_decays,num_hiddens,patiences

if __name__=="__main__":
    deploy_path = get_path.FM_ParametersDeploy_path
    label = "FMcross"
    parameter_deploy = get_FMparameters_deploy()
    ParametersFind(label,parameter_deploy,train)
    # ParametersTest(label,parameter_deploy,train)

    
