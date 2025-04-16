from tqdm import tqdm
from traditon_utils import compute_sim,get_data,evaluate
from collections import defaultdict

class UserIIF:
    def __init__(self,user_item,ppl_user):
        print("正在进行协同过滤UserIIF推荐")
        self.user_item = user_item
        self.ppl_user = ppl_user

    def get_user_sim(self,k):
        sim_dict = {}
        qbar = tqdm(total=len(self.user_item),desc="正在生成相似用户表")
        for u1,u1_item in self.user_item.items():
            sim_k_list = []
            for u2,u2_item in self.user_item.items():
                if u1==u2 or len(u1_item&u2_item)==0:
                    continue
                sim = compute_sim(u1_item,u2_item,self.ppl_user,u1)
                sim_k_list.append((u2,sim))
            sim_k_list = [j[0] for j in sorted(sim_k_list,key=lambda x:x[1],reverse=True)[:k]]
            sim_dict[u1] = sim_k_list
            qbar.update(1)
        return sim_dict


    def get_recommand(self,user_sim):
        recommand_dict = defaultdict(set)
        qbar = tqdm(total=len(user_sim),desc="正在生成推荐表")
        for u1 in user_sim:
            u1_item = self.user_item[u1]
            for u1_sim in user_sim[u1]:
                recommand_dict[u1] |= self.user_item[u1_sim] - u1_item
            qbar.update(1)
        return recommand_dict
    

def main():
    train_data,test_df = get_data(test_ratio=0.5)
    iif = UserIIF(*train_data[0])
    user_sim = iif.get_user_sim(5)
    user_rec = iif.get_recommand(user_sim)
    evaluate(test_df,user_rec)

if __name__=="__main__":
    main()