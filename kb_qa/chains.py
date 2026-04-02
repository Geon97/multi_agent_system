from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import _get_langchain_llm

# LangChain 基于上下文生成答案
def build_answer_chain(llm=None):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个个人知识库问答助手。"
                "只允许基于提供的知识库上下文回答，不要编造事实。"
                "优先使用标记为“直接证据”的片段回答。"
                "如果没有直接证据，但有多条“间接证据”能支撑时间或先后关系，可以做弱推断。"
                "弱推断必须明确写成“未见明确日期，但结合现有片段可推断……”，不能把推断说成确定事实。"
                "不要把规划编制时间直接等同于整改完成时间；如果只能说明先后关系，就只能回答先后关系。"
                "只有在直接证据和间接证据都不足时，才回答“根据当前知识库无法确认”。"
                "回答使用中文，先给结论，再给简短依据；若使用弱推断，请单独写“推断说明”。"
                "如果引用了上下文，请在句子末尾使用 [1]、[2] 这样的编号引用。",
            ),
            (
                "human",
                "用户问题:\n{question}\n\n"
                "证据概览:\n{evidence_summary}\n\n"
                "知识库上下文:\n{context}\n\n"
                "请输出最终回答。",
            ),
        ]
    )
    return prompt | (llm or _get_langchain_llm()) | StrOutputParser()
