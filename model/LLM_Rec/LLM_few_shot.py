from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


class LLM_FewShot:
    def __init__(self, prompt_prefix, prompt_suffix, variables):
        OPENAI_API_KEY = "sk-8FHanPyzQG29mIrm125eDf81B0Aa468d93Fd8fCd0b4356Ef"
        OPENAI_API_BASE = "http://openai-proxy.miracleplus.com/v1"
        llm = ChatOpenAI(model='gpt-4o', api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        
        # 定义示例
        examples = [
            {
                "sequeence": '（"视频id":10595,"封面文字":印度军队再次非法越线,"简介标题":印度军队再次非法越线,"标题标签":[],"一级标签":军事,"二级标签":军事冲突,"三级标签":UNKNOWN）->（"视频id":6502,"封面文字":UNKNOWN,"简介标题":#谭松韵庭审现场哽咽发言 真的令人心疼…庭审后紧紧与亲友相拥 ,"标题标签":[谭松韵庭审现场哽咽发言],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":社会新闻）->（"视频id":4944,"封面文字":UNKNOWN,"简介标题":开了11年的车，还不知道安全带原有这种用途，受教了！ #主播中心 #快手创作者中心 ,"标题标签":[主播中心,快手创作者中心],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":社会新闻）->（"视频id":4987,"封面文字":UNKNOWN,"简介标题":辽宁大量特警突袭一药店，多人蒙头套被押！目击者：该药店开业才几个月，看起来很普通,"标题标签":[],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":法制事件）->（"视频id":1643,"封面文字":UNKNOWN,"简介标题":#高清视频 #幼儿园开学翻车现场 睡午觉到底是找爸爸还是找奶奶呢？你俩让妈妈怎么想😂,"标题标签":[幼儿园开学翻车现场,高清视频],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":趣闻趣事）->（"视频id":7975,"封面文字":UNKNOWN,"简介标题":另类“生活小妙招”,"标题标签":[],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":趣闻趣事）->（"视频id":1667,"封面文字":UNKNOWN,"简介标题":蒙冤入狱9778天，张玉环申请国家赔偿2234万元，要求法院道歉！,"标题标签":[],"一级标签":民生资讯,"二级标签":社会事件,"三级标签":社会新闻）->（"视频id":3382,"封面文字":UNKNOWN,"简介标题":这保姆还能要不,"标题标签":[],"一级标签":亲子,"二级标签":晒娃,"三级标签":4至12岁萌娃）->（"视频id":4712,"封面文字":UNKNOWN,"简介标题":有一种“屁股大”叫“假胯宽”，这个动作教你练成迷人身材！ #瑜伽 ,"标题标签":[瑜伽],"一级标签":健身,"二级标签":放松训练,"三级标签":瑜伽）->（"视频id":5027,"封面文字":UNKNOWN,"简介标题":你知道他怎么回去的吗？ #跳水 #轻松一刻,"标题标签":[跳水,轻松一刻],"一级标签":运动,"二级标签":UNKNOWN,"三级标签":UNKNOWN）', 
                "candidate": '（视频id:7049,  封面文字:UNKNOWN,  简介标题:#篮球 easy bro  #扣篮 #球场工具人 ,  标题标签:[扣篮,球场工具人,篮球],  一级标签:运动,  二级标签:球类运动,  三级标签:篮球）',
                "output": "0.67"
            }
        ]
        
        # 示例模板
        example_template = """
                        历史序列: {sequeence}
                        候选视频: {candidate}
                        预测点击率: {output}
                        """
        
        example_prompt = PromptTemplate(
            input_variables=["sequeence", "candidate", "output"],
            template=example_template
        )
        
        # 创建 few-shot 提示模板
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            input_variables=variables
        )
        
        self.llm_chain = few_shot_prompt | llm

    def FewShot(self, sequeence, candidate):
        result = self.llm_chain.invoke({
            "sequeence": str(sequeence),
            "candidate": str(candidate)
        })
        return result


# 使用示例
if __name__ == "__main__":
    prompt_prefix = """你是一个视频推荐系统。根据用户观看历史，预测用户点击新视频的概率。
    """
    prompt_suffix = """
    历史序列: {sequeence}
    候选视频: {candidate}

    请根据上述历史序列，预测用户点击候选视频的概率。
    返回一个0到1之间的浮点数，只返回数字，不要有任何解释。
    """
    
    llm_few_shot = LLM_FewShot(prompt_prefix, prompt_suffix, ["sequeence", "candidate"])
    
    # 测试示例
    user_history = "{'视频id': 10595, '封面文字': '印度军队再次非法越线'...}" # 这里放实际历史
    candidate_video = "{视频id:7049, 封面文字:UNKNOWN...}" # 这里放候选视频
    
    prediction = llm_few_shot.FewShot(user_history, candidate_video)
    print(f"预测点击率: {prediction.content}")

