from model.LLM_Rec.LLM_few_shot import LLM_FewShot
from load_data.LLM_utils import get_video_prompt
import pickle
import json
from load_data import get_path
from collections import defaultdict
import concurrent
from tqdm import tqdm

prompt_prefix = """你是一个视频推荐系统。根据用户观看历史，预测用户点击新视频的概率。
"""
prompt_suffix = """
历史序列: {sequeence}
候选视频: {candidate}

请根据上述历史序列，预测用户点击候选视频的概率。
返回一个0到1之间的浮点数，只返回数字，不要有任何解释。
"""


def get_response(llm,sequeence,candidate):
    for _ in range(10):
        response = llm.FewShot(sequeence,candidate)
        try:
            response = eval(response.content)
            break
        except:
            response = None
            continue
    return response

def process_candidate(args):
    llm, sequence, v, candidate_dict = args
    if str(v) not in candidate_dict.keys():
        return None
    single_candidate = ''
    single_candidate += candidate_dict[str(v)] + "\n"
    response = get_response(llm, sequence, single_candidate)
    if response is not None:
        return (v, response)
    return None

def get_llm_sequence_recommend():
    llm = LLM_FewShot(prompt_prefix,prompt_suffix,["sequeence","candidate"])
    prompt_user_dict = get_video_prompt()
    recommend_dict = pickle.load(open(get_path.recommend_dict_path,"rb"))
    candidate_dict = json.load(open(get_path.candidate_video_describe_path))
    llm_recommend_dict = defaultdict(list)
    
    # 设置线程池最大工作线程数
    max_workers = 15
    
    for u in tqdm(recommend_dict):
        if u not in prompt_user_dict.keys():
            continue
        sequence = prompt_user_dict[u]
        candidate_video = recommend_dict[u]
        output = []
        
        # 准备参数列表
        tasks = [(llm, sequence, v, candidate_dict) for v in candidate_video if v in candidate_video.keys()]
        
        # 使用多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_candidate, tasks))
            
        # 过滤掉None结果并排序
        output = [result for result in results if result is not None]
        output = [i[0] for i in sorted(output, key=lambda x:x[1], reverse=True)]
        output = output[:5] if len(output)>=5 else output
        llm_recommend_dict[u].append(output)
    
    json.dump(llm_recommend_dict, open(get_path.llm_recommend_dict_path, "w+", encoding="utf-8"), indent=4)