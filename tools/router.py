from config import get_llama_index_llm
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool


def build_router_source(query_engine, name, description):
    """
    为 LlamaIndex Router 构建候选查询源。

    注意这里返回的仍然是 LlamaIndex 自己的 `QueryEngineTool`，
    因为 `RouterQueryEngine` 需要这套对象来完成候选源选择。
    """

    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name=name,
        description=description,
    )


def build_router_engine(query_engine_tools, llm=None):
    """
    构建统一路由引擎。

    这里必须显式注入 LLM，不能依赖 `RouterQueryEngine.from_defaults()`
    内部回退到 `Settings.llm`，否则 LlamaIndex 会走自己的默认 OpenAI 模型，
    通常就是 `gpt-3.5-turbo`。
    """

    return RouterQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        llm=llm or get_llama_index_llm(),
    )
