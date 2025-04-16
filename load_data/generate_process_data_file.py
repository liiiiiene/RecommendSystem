import load_data.process_utils as pre_process
import load_data.FM_utils as FM_process
import load_data.bert_utils as gpt_embedding_process
# from model.EmbeddingLayer.GptEmbedding import main as get_gpt_embedding
import load_data.WideAndDeep_utils as model_process 
import load_data.seqs_uilts as seqs_process
import load_data.LLM_utils as llm_process
import os

if __name__=="__main__":
    os.makedirs("process_data",exist_ok=True)
    # 测试数据 pre_process.valid_small_martix()
    print("正在生成预处理数据")
    pre_process.valid_small_martix()
    # 生成三元组(u,i,r) .csv文件
    pre_process.get_triple() # "process_data/triples.csv"
    # 生成item特征索引文件
    pre_process.get_item_df() # "process_data/item_df.csv"
    # 生成user特征索引列表文件
    pre_process.get_user_df() # "process_data/user_comsum.csv" / "process_data/user_df.csv"
    # 从三元组(u,i,r) .csv文件 拆分 训练集和测试集三元组
    pre_process.split_train_test() # "process_data/test_df.csv" / "process_data/train_df.csv"


    print("正在生成FMcross需要的数据")
    # FMcross 需要的文件生成
    # 处理item 和user 的特征值，缺失值用-1填充，统计所有的 item和user 数量
    FM_process.get_save_feature() # "load_data/deployment.json" / "process_data/item_feature.csv" / "process_data/user_feature.csv"
    # # 获得视频的标题标签等描述性信息
    gpt_embedding_process.get_item_title() # "process_data/video_text.csv"

    # # 获得视频 embedding 数据
    # 已经生成，直接上传
    # get_gpt_embedding() # "process_data/gpt_embeddings.csv"

    
    # # 提取出embedding 数据，使用 torch.tensor 格式储存
    FM_process.tensor_embedding() # "process_data/gpt_emb.pt"


    print("正在生成Transformers需要的数据")
    # Transformers 需要的文件生成
    # 生成序列和用户序列
    seqs_process.get_seqs()
    # "process_data/user_seq.json" 用户——序列
    # "process_data/seq_item_rato.npy" 视频序列
    # "load_data/deployment.json" 获得物品的最大索引

    print("正在生成组合模型需要的数据")
    model_process.build_user_seqs() # "process_data/user_seq_ratio.npy"

    print("生成视频prompt")
    llm_process.build_video_prompt()
    llm_process.build_candidate_prompt()
