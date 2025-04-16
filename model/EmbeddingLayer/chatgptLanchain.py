from langchain_openai import OpenAIEmbeddings

class DeployLangChain:
    """LangChain基础部署类，支持文本生成和嵌入"""
    
    def __init__(self,embedding_model='text-embedding-3-small'):
        # 优先使用环境变量中的配置
        self.api_key = "sk-8FHanPyzQG29mIrm125eDf81B0Aa468d93Fd8fCd0b4356Ef"
        self.api_base = "http://openai-proxy.miracleplus.com/v1"
        
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key,
            base_url=self.api_base
        )

    def batch_embed(self, texts, batch_size=128):
        """安全批量嵌入方法"""
        if not self.embeddings:
            raise ValueError("嵌入模型未正确初始化")
        
        total = len(texts)
        results = []
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            try:
                embeddings = self.embeddings.embed_documents(batch)
            except Exception as e:
                print(f"第 {i//batch_size+1} 批处理失败: {str(e)}")
                embeddings = [None]*batch
            finally:
                print(f"处理进度: {min(i+batch_size, total)}/{total} ({((i+batch_size)/total)*100:.1f}%)")

            results.extend(embeddings)

        return results