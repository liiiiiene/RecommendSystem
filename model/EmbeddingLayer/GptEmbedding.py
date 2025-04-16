from model.EmbeddingLayer.chatgptLanchain import DeployLangChain
from load_data import get_path
import pandas as pd



class GptEmbedding(DeployLangChain):
    def __init__(self, data):
        super().__init__()  # 初始化父类LangChain配置
        self.data = data
    
    def generate_embeddings(self):
        """优化后的批量嵌入方法"""
        texts = self.data['text'].tolist()
        return self.batch_embed(texts)

def main():
    # 读取预处理后的文本数据
    data = pd.read_csv(get_path.video_title_path, index_col=0)

    process_data = data[pd.isna(data["embedding"])]
    processed_data = data[~process_data.index.isin(data.index)]
    # 初始化嵌入生成器
    embedder = GptEmbedding(process_data)
    
    # 生成并保存嵌入
    embeddings = embedder.generate_embeddings()
    process_data['embedding'] = embeddings
    total_data = pd.concat([process_data,processed_data],axis=0)
    total_data.to_csv(get_path.gpt_embeddings_path)

if __name__ == "__main__":
    main()

