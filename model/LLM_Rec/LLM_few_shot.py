from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class LLM_FewShot:
    def __init__(self,prompt_text,variables):
        OPENAI_API_KEY = "sk-8FHanPyzQG29mIrm125eDf81B0Aa468d93Fd8fCd0b4356Ef"
        OPENAI_API_BASE = "http://openai-proxy.miracleplus.com/v1"
        llm = ChatOpenAI(model='gpt-4o', api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        prompt = PromptTemplate(
            input_variables=variables,
            template=prompt_text
        )
        self.llm_chain = prompt | llm

    def FewShot(self):
        pass

