import json
from load_data import get_path,LLM_utils
from model.LLM_Rec.LLM_few_shot import LLM_explain
import concurrent
from threading import Lock
from tqdm import tqdm
from collections import defaultdict

prompt_prefix = """
你是一个视频推荐系统。根据用户观看历史序列，对新推荐的视频进行解释说明推荐这个视频的原因。
"""
prompt_suffix = """
历史序列：{sequeence}
推荐的视频:{recommend_video}

请根据上述历史序列以及推荐视频的描述，对新推荐的视频进行解释，说明为什么要推荐这个视频？用一段话概括性描述
"""
def get_response(llm,sequeence,recommend_video):
    for i in range(10):
        try:
            return llm.ExplainRec(sequeence,recommend_video).content
        except:
            continue

def process_video(args):
    llm,prompt_seqs,v,video_describe_dict = args
    item_explain = dict()
    item_des = video_describe_dict[str(v)]
    response = get_response(llm,prompt_seqs,item_des)
    item_explain[str(v)] = response
    return item_explain

def process_user(args):
    u, rec_list, prompt_seqs, video_describe_dict, llm = args
    max_workers = 5
    user_explain_dict = {}
    
    tasks = [(llm, prompt_seqs, v, video_describe_dict) for v in rec_list]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_video, tasks))
    
    for item_explain in results:
        user_explain_dict.update(item_explain)
    
    return u, user_explain_dict

def user_explain_llm_recommend_dict():
    llm_recommend_dict = json.load(open(get_path.llm_recommend_dict_path, "r+", encoding="utf-8"))
    prompt_user_seqs = LLM_utils.get_video_prompt()
    video_describe_dict = LLM_utils.get_candidate_prompt()
    llm = LLM_explain(prompt_prefix, prompt_suffix, ["sequeence", "recommend_video"])
    
    max_workers = 15  # 用户级别的最大线程数
    explain_recommend_dict = defaultdict(dict)
    
    tasks = [(u, llm_recommend_dict[u][0], prompt_user_seqs[u], video_describe_dict, llm) 
            for u in llm_recommend_dict]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for u, user_explain_dict in tqdm(executor.map(process_user, tasks), 
                                        total=len(tasks), 
                                        desc="正在生成对推荐列表的解释"):
            explain_recommend_dict[u].update(user_explain_dict)
    
    with Lock():
        json.dump(explain_recommend_dict, open(get_path.explain_llm_recommend_dict_path, "w+", encoding="utf-8"),indent=4,ensure_ascii=False)


if __name__=="__main__":
    user_explain_llm_recommend_dict()



# def item_explain_llm_recommend_dict():
#     llm_recommend_dict = json.load(open(get_path.llm_recommend_dict_path,"r+",encoding="utf-8"))
#     prompt_user_seqs = LLM_utils.get_video_prompt()
#     video_describe_dict = LLM_utils.get_candidate_prompt()
#     llm = LLM_explain(prompt_prefix,prompt_suffix,["sequeence","recommend_video"])

#     max_workers = 5
#     explain_recommend_dict = defaultdict(dict)
#     for u in tqdm(llm_recommend_dict,desc="正在生成对推荐列表的解释"):
#         rec_list = llm_recommend_dict[u][0]
#         prompt_seqs = prompt_user_seqs[u]
        
#         tasks = [(llm , prompt_seqs, v , video_describe_dict) for v in rec_list]

#         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as excuter:
#             results = list(excuter.map(process_video,tasks))

#         item_dict = dict()
#         for item_explain in results:
#             item_dict.update(item_explain)
#         explain_recommend_dict[u].update(item_dict)

#     with Lock():
#         json.dump(explain_recommend_dict,open(get_path.explain_llm_recommend_dict_path,"w+",encoding="utf-8"))