from langchain_community.embeddings import BaichuanTextEmbeddings
import pandas as pd
from load_data import get_path
from utils import random_time

class baichuanLangChain:
    def __init__(self,data):
        self.api_key = "sk-5d2efc453e8a92063f498c8840abe437"
        self.data = data

        self.embedding = BaichuanTextEmbeddings(
        baichuan_api_key = self.api_key
        )
    def batch_embedding(self,text,batch_size=16):
        result = []
        total = len(text)
        for i in range(0,total,batch_size):

            batch = text[i:i+batch_size]
            try:
                batch_emb = self.embedding.embed_documents(batch)
                print(f"目前进度：{min(i+batch_size,total)}/{total} ({(i+batch_size/total)*100:.1f}%)")
            except:
                print(f"error in batch {i+1}")
                batch_emb = [None]*batch_size

            result.extend(batch_emb)
            random_time(30,40)

        return result
        

    def generate_embeddings(self):
        text = self.data['text'].tolist()
        return self.batch_embedding(text)
    
if __name__ == "__main__":
    # 读取预处理后的文本数据
    data = pd.read_csv(get_path.video_title_path, index_col=0)

    process_data = data[pd.isna(data["embedding"])]
    processed_data = data[~process_data.index.isin(data.index)]
    # 初始化嵌入生成器
    embedder = baichuanLangChain(process_data)
    
    # 生成并保存嵌入
    embeddings = embedder.generate_embeddings()
    process_data['embedding'] = embeddings
    total_data = pd.concat([process_data,processed_data],axis=0)
    total_data.to_csv(get_path.baichuan_embddings_path)
