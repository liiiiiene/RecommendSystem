from model.Deep.Transformer4Rec import Transformer4Rec
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from load_data.seqs_uilts import load_seqs
from torch.utils.data import DataLoader
from utils import Accumulator
from load_data import get_path
from utils.utils import SaveModel,BestModel,EarlyStop
from sklearn.metrics import precision_score,recall_score,accuracy_score
from tqdm import tqdm
import json
from utils.multi_gpu_train import ParametersFind,ParametersTest


def doeval(test_data,net,batch_size,value,device):
    net.eval()
    metric = Accumulator(4)
    for seqs in DataLoader(test_data,batch_size=batch_size,shuffle=True):

        history_sequence = torch.LongTensor(seqs[:,:-2].numpy()).to(device)
        target_sequence = torch.LongTensor(seqs[:,1:-1].numpy()).to(device)
        target_item = torch.LongTensor(seqs[:,-2].numpy()).to(device)
        target_labels = torch.FloatTensor(seqs[:,-1].numpy())


        Rec_logits,_ = net(history_sequence,target_sequence,target_item)
        logits = (Rec_logits >= value).type(torch.int).to("cpu")
        precicion = precision_score(target_labels,logits,zero_division=0)
        recall = recall_score(target_labels,logits,zero_division=0)
        accuracy = accuracy_score(target_labels,logits)

        metric.add(precicion,recall,accuracy,1)
    net.train()
    return metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3]

# num_epochs,lr,batch_size,weight_decay,num_hidden,patience,device,net_path,predict_path,parameter_path
def train(num_epochs, lr, batch_size, weight_decay, num_hidden, patience, 
        num_heads,num_layer,dropout,device,net_path,predict_path,parameter_path):
    TrainParameter = {
            "transformer":{
                "num_epochs": num_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "num_hidden": num_hidden,
                "patience": patience,
                "num_head": num_heads,
                "num_layer":num_layer,
                "dropout":dropout
                }
    }

    train_data,test_data,n_item = load_seqs()
    all_seq_len = train_data.shape[1] - 1
    net = Transformer4Rec(n_item,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden*2,
                        all_seq_len,num_heads,num_layer,dropout)

    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)

    # 最佳模型
    net_save = Transformer4Rec(n_item,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden*2,
                        all_seq_len,num_heads,num_layer,dropout)
    best_model = BestModel(net_path,predict_path,parameter_path,"transformer",net_save,device)

    save_model = SaveModel(net_path,predict_path,parameter_path,"transformer")

    early_stop = EarlyStop(patience)

    scaler = torch.amp.GradScaler()
    

    qbar = tqdm(range(num_epochs),
            desc=f"batch_size:{batch_size},lr:{lr},weight_decay:{weight_decay},num_hidden:{num_hidden},patience:{patience},dropout:{dropout},num_heads:{num_heads},num_layer{num_layer}")
    for e in qbar:
        metric = Accumulator(2)
        for seqs in DataLoader(train_data,batch_size,shuffle=True):
            torch.cuda.empty_cache()
            history_sequence = torch.LongTensor(seqs[:,:-2].numpy()).to(device)
            target_sequence = torch.LongTensor(seqs[:,1:-1].numpy()).to(device)
            target_item = torch.LongTensor(seqs[:,-2].numpy()).to(device)
            target_labels = torch.FloatTensor(seqs[:,-1].numpy()).to(device)  # BCEloss需要float类型
            optimizer.zero_grad()
            with torch.amp.autocast(device):
                Rec_logit,Seq_out = net(history_sequence,target_sequence,target_item)
                loss = net.compute_loss(Rec_logit,target_labels,Seq_out,target_sequence)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            metric.add(loss.item(),2)

        precision,recall,accuracy = doeval(test_data,net,batch_size,0.5,device)

        # 检查是否有改善
        if early_stop.check(best_model.isBest(metric[0]/metric[1],precision,recall,accuracy,net,TrainParameter)):
            print(f"早停：连续{patience}个epoch没有改善，在第{e+1}个epoch停止训练")
            break
        
        qbar.set_postfix({
            "best_loss": f"{best_model.best_loss:.4f}",
            "best_precision": f"{best_model.best_precision:.4f}"
        })
        

    save_model.save_model(best_model)

# num_epochs, lr, batch_size, weight_decay, num_hidden, patience, num_heads,num_layer,dropout
def ParametersDeploy():
    optional_parameter = json.load(open(get_path.Transformer_ParameterDeploy_path))
    lrs = optional_parameter["lr"]
    batch_sizes = optional_parameter["batch_size"]
    weight_decays = optional_parameter["weight_decay"]
    num_hiddens = optional_parameter["num_hidden"]
    patiences = optional_parameter["patience"]
    num_epochs = optional_parameter["num_epochs"]
    num_layers = optional_parameter["num_layer"]
    num_heads = optional_parameter["num_head"]
    dropouts = optional_parameter["dropout"]
    return num_epochs,lrs,batch_sizes,weight_decays,num_hiddens,patiences,num_heads,num_layers,dropouts

if __name__=="__main__":
    label = "transformer"
    parameter_deploy = ParametersDeploy()
    ParametersFind(label,parameter_deploy,train)
    # ParametersTest(label,parameter_deploy,train)
    