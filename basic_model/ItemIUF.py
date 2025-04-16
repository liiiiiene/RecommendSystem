from traditon_utils import get_data,compute_sim,evaluate
from tqdm import tqdm
from collections import defaultdict

class ItemIUF:
    def __init__(self,item_user,ppl_item):
        print("正在进行协同过滤推荐ItemIUF")
        self.item_user,self.ppl_item = item_user,ppl_item


    def get_item_sim(self,k):
        item_sim = {}
        qbar = tqdm(total=len(self.item_user),desc="正在生成相似物品表")
        for i1,i1_u in self.item_user.items():
            item_k_list = []
            for i2,i2_u in self.item_user.items():
                if i1==i2 or len(i1_u&i2_u)==0:
                    continue
                sim = compute_sim(i1_u,i2_u,self.ppl_item,i1)
                item_k_list.append((i2,sim))
            item_k_list = [j[0] for j in sorted(item_k_list,key=lambda x:x[1],reverse=True)[:k]]
            item_sim[i1] = item_k_list
            qbar.update(1)
        return item_sim
    

    def get_recommand(self,user_item,item_sim):
        recommand_dict = defaultdict(set)
        qbar = tqdm(total=len(self.item_user),desc="正在生成推荐列表")
        for u in user_item:
            u_item =  user_item[u]
            for i in u_item:
                recommand_dict[u] |= set(item_sim[i]) - u_item
            qbar.update(1)
        return recommand_dict


def main():
    train_data,test_df = get_data(test_ratio=0.5)
    iuf = ItemIUF(*train_data[1])
    item_sim = iuf.get_item_sim(5)
    recommand_dict = iuf.get_recommand(train_data[0][0],item_sim)
    evaluate(test_df,recommand_dict)

if __name__=="__main__":
    main()


