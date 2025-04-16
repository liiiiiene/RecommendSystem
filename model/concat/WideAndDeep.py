from model.Deep.Transformer4Rec import Transformer4Rec
from model.Wide.FeatureCross import FeatureCross
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
from model.Deep.Transformer4Rec import Transformer4Rec
import torch
from torch.utils.data import DataLoader
from load_data import FM_utils
from load_data.WideAndDeep_utils import get_user_seqs
from utils.utils import Accumulator
from load_data import get_path
from utils.utils import SaveModel,BestModel,EarlyStop
from sklearn.metrics import precision_score,recall_score,accuracy_score
from tqdm import tqdm
import json
from utils.multi_gpu_train import ParametersFind,ParametersTest


class WideAndDeep(nn.Module):

    def __init__(self,n_item,query_size,key_size,value_size,num_hidden,
                fnn_input,fnn_hidden,all_seqs_len,num_heads,num_layer,dropout,
                user_feature, item_feature, gpt_embedding, 
                n_user_feature, n_item_feature, device,**kwargs):
        super().__init__(**kwargs)
        self.transformer = Transformer4Rec(n_item,query_size,key_size,value_size,
                                        num_hidden,fnn_input,fnn_hidden,all_seqs_len,
                                        num_heads,num_layer,dropout)
        
        self.fmcross = FeatureCross(user_feature,item_feature,gpt_embedding,
                                    n_user_feature,n_item_feature,num_hidden,device)
        
        # 两层残差
        self.resnet1 = self.__residual(1,num_hidden,num_hidden)
        self.reshapeX1 = nn.Conv1d(1,num_hidden,kernel_size=1)
        self.resnet2 = self.__residual(num_hidden,num_hidden*2,num_hidden)
        self.reshapeX2 = nn.Conv1d(num_hidden,num_hidden*2,kernel_size=1)
        self.mlp = self.__mlp(num_hidden*2)

        self.project = nn.Linear(1,num_hidden)

    def Logit(self,X):
        y = self.resnet1(X)
        y += nn.functional.relu(self.reshapeX1(X))
        y1 = self.resnet2(y)
        y1 += nn.functional.relu(self.reshapeX2(y))
        y1 = self.mlp(y1)
        return y1
    
    def __residual(self,input_dim,output_dim,normal_shape):
        return nn.Sequential(
            nn.Conv1d(input_dim,output_dim,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.LayerNorm([output_dim,normal_shape]),
            nn.Conv1d(output_dim,output_dim,kernel_size=3,padding=1,stride=1),
            nn.LayerNorm([output_dim,normal_shape])
        )
    
    def __mlp(self,input):
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input,1),
            nn.Sigmoid()
        )

    def forward(self,uid,history_seq,target_seq,target_item):
        fm_logits = self.fmcross(uid,target_item)
        tran_logits,tran_seq = self.transformer(history_seq,target_seq,target_item)
        logits = self.Logit(self.project((fm_logits + tran_logits).reshape(-1,1)).unsqueeze(dim=1))
        return logits.squeeze(dim=1),tran_seq
    
    def loss(self,logits,target_label,seqs_out,target_seq):
        return self.transformer.compute_loss(logits,target_label,seqs_out,target_seq)

def doeval(test_data,net,batch_size,value,device):
    net.eval()
    metric = Accumulator(4)
    for seqs in DataLoader(test_data,batch_size=batch_size):

        uid = torch.LongTensor(seqs[:,0].numpy())
        history_sequence = torch.LongTensor(seqs[:,1:-2].numpy()).to(device)
        target_sequence = torch.LongTensor(seqs[:,2:-1].numpy()).to(device)
        target_item = torch.LongTensor(seqs[:,-2].numpy()).to(device)
        target_labels = torch.FloatTensor(seqs[:,-1].numpy())

        Rec_logits,_ = net(uid,history_sequence,target_sequence,target_item)
        logits = (Rec_logits >= value).type(torch.int).to("cpu")
        precicion = precision_score(target_labels,logits,zero_division=0)
        recall = recall_score(target_labels,logits,zero_division=0)
        accuracy = accuracy_score(target_labels,logits)

        metric.add(precicion,recall,accuracy,1)
    net.train()
    return metric[0] / metric[3], metric[1] / metric[3], metric[2] / metric[3]


def train(num_epochs, lr, batch_size, weight_decay, num_hidden, patience,
        num_heads,num_layer,dropout,device,net_path,predict_path,parameter_path):

    TrainParameter = {
        "WideAndDeep":{
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
    
    item_feature,user_feature,num_item_feature,num_user_feature = FM_utils.get_feature()
    gpt_embedding = FM_utils.get_emb_tensor()
    n_item,train_data,test_data = get_user_seqs()
    all_seq_len = train_data.shape[1] - 2
    net = WideAndDeep(n_item,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden*2,
                        all_seq_len,num_heads,num_layer,dropout,user_feature,item_feature,
                        gpt_embedding,num_user_feature,num_item_feature,device)
    net.to(device)
    net_save = WideAndDeep(n_item,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden,num_hidden*2,
                        all_seq_len,num_heads,num_layer,dropout,user_feature,item_feature,
                        gpt_embedding,num_user_feature,num_item_feature,device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)

    # 最佳模型
    best_model = BestModel(net_path,predict_path,parameter_path,"WideAndDeep",net_save,device)

    save_model = SaveModel(net_path,predict_path,parameter_path,"WideAndDeep")

    early_stop = EarlyStop(patience)

    scaler = torch.amp.GradScaler()  # 新增：混合精度梯度缩放

    qbar = tqdm(range(num_epochs),
            desc=f"batch_size:{batch_size},lr:{lr},weight_decay:{weight_decay},num_hidden:{num_hidden},patience:{patience},dropout:{dropout},num_heads:{num_heads},num_layer{num_layer}")

    for e in qbar:
        metric = Accumulator(2)
        for seqs in DataLoader(train_data,batch_size,shuffle=True):
            torch.cuda.empty_cache()
            uid = torch.LongTensor(seqs[:,0].numpy())
            history_sequence = torch.LongTensor(seqs[:,1:-2].numpy()).to(device)
            target_sequence = torch.LongTensor(seqs[:,2:-1].numpy()).to(device)
            target_item = torch.LongTensor(seqs[:,-2].numpy()).to(device)
            target_labels = torch.FloatTensor(seqs[:,-1].numpy()).to(device)  # BCEloss需要float类型
            optimizer.zero_grad()
            with torch.amp.autocast(device):
                logit,tran_seq = net(uid,history_sequence,target_sequence,target_item)
                loss = net.loss(logit,target_labels,tran_seq,target_sequence)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
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
        

    save_model.save_model(best_model)  # 而不是保存整个模型

# num_epochs, lr, batch_size, weight_decay, num_hidden, patience, num_heads,num_layer,dropout
def ParametersDeploy():
    optional_parameter = json.load(open(get_path.WideAndDeep_ParameterDeploy_path))
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
    parameter_deploy = ParametersDeploy()
    label = "WideAndDeep"
    ParametersFind(label,parameter_deploy,train)
    # ParametersTest(label,parameter_deploy,train)

    # test
    # num_epochs,lrs,batch_sizes,weight_decays,num_hiddens,patiences,num_heads,num_layer,dropout = 2,0.001,1024,0.001,120,10,6,4,0.7
    # device = "cuda"
    # net_path,predict_path,parameter_path = get_path.WideAndDeep_net_path,get_path.WideAndDeep_BestPredict_path,get_path.WideAndDeep_BestParameter_path
    # parameter_deploy = (num_epochs,lrs,batch_sizes,weight_decays,num_hiddens,patiences,num_heads,num_layer,dropout,device,net_path,predict_path,parameter_path)
    # train(*parameter_deploy)

