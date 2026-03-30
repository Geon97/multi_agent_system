import os
from typing import Any

from crewai import LLM as CrewAILLM
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import SecretStr
from qdrant_client import QdrantClient

load_dotenv()


def _get_env(name: str, default: str | None = None) -> str | None:
    """
    读取环境变量，并顺手做一层轻量清洗。

    这里主要处理几类常见情况：
    1. 变量存在但为空字符串
    2. 变量前后被额外加了空格
    3. `.env` 里把值写成了带引号的形式
    """

    value = os.getenv(name)
    if value is None:
        return default

    cleaned = value.strip().strip('"').strip("'")
    if cleaned == "":
        return default
    return cleaned


api_key = _get_env("OPENAI_API_KEY")
api_base = _get_env("OPENAI_API_BASE")
llm_model = _get_env("OPENAI_MODEL_NAME")
embedding_api_base = _get_env("OPENAI_EMBED_BASE", api_base)
embedding_api_key = _get_env("OPENAI_EMBED_KEY", api_key)
embedding_model = _get_env("OPENAI_EMBED_MODEL", "bge-m3")
llm_provider = _get_env("LLM_PROVIDER", "openai")
crewai_provider = _get_env("CREWAI_PROVIDER", llm_provider or "openai")
crewai_model = _get_env("CREWAI_MODEL_NAME", llm_model)
qdrant_api_key = _get_env("QDRANT_API_KEY")
qdrant_url = _get_env("QDRANT_URL")

# 第三方 OpenAI 兼容模型往往不是 LlamaIndex 内置白名单中的模型名，
# 因此这里允许通过环境变量显式补充上下文窗口和输出 token 上限。
llama_index_context_window = int(_get_env("LLAMA_INDEX_CONTEXT_WINDOW", "32768"))
llama_index_num_output = int(_get_env("LLAMA_INDEX_NUM_OUTPUT", "2048"))

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if api_base:
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base
if llm_model:
    os.environ["OPENAI_MODEL_NAME"] = llm_model
if llm_provider:
    os.environ["LLM_PROVIDER"] = llm_provider


def _extract_text_from_langchain_response(response: Any) -> str:
    """把 LangChain 返回对象统一整理成纯文本。"""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)


def _get_langchain_llm() -> ChatOpenAI:
    """
    给 LlamaIndex 侧使用的 LangChain LLM。

    这里保留 `ChatOpenAI`，因为它对第三方 OpenAI 兼容网关更宽松，
    不会像 LlamaIndex 自带的 OpenAI LLM 一样先在本地校验模型名。
    """

    return ChatOpenAI(
        model=llm_model,
        temperature=0.3,
        api_key=SecretStr(api_key) if api_key else None,
        base_url=api_base,
    )


class CompatibleLlamaIndexLLM(CustomLLM):
    """
    用 LangChain 的 `ChatOpenAI` 适配出一个 LlamaIndex 可用的 LLM。

    这样做是为了兼容第三方 OpenAI 兼容接口上的自定义模型名，
    例如 `Qwen/Qwen3-8B`、`deepseek-v3.1` 等。
    """

    model_name: str = Field(default="unknown")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=2048)
    _llm: ChatOpenAI = PrivateAttr()

    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        model_name: str,
        context_window: int,
        num_output: int,
    ) -> None:
        super().__init__(
            model_name=model_name,
            context_window=context_window,
            num_output=num_output,
        )
        self._llm = llm

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            is_chat_model=False,
            is_function_calling_model=False,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = self._llm.invoke(prompt)
        return CompletionResponse(
            text=_extract_text_from_langchain_response(response),
            raw=response,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        # 当前项目没有依赖真正的流式合成，这里返回单次结果即可。
        yield self.complete(prompt, formatted=formatted, **kwargs)


def get_llm() -> CrewAILLM:
    """
    返回 CrewAI 可直接接受的 LLM 实例。

    关键点：
    - 不能再返回 LangChain `ChatOpenAI`，否则 `Agent.llm` 会校验失败
    - 对于第三方 OpenAI 兼容网关，优先按 `openai` 协议访问
    - 如有特殊 provider，再通过 `CREWAI_PROVIDER` 覆盖
    """

    return CrewAILLM(
        model=crewai_model or llm_model or "gpt-4o-mini",
        provider=crewai_provider or "openai",
        temperature=0.3,
        api_key=api_key,
        base_url=api_base,
        api_base=api_base,
    )


def get_llama_index_llm() -> CompatibleLlamaIndexLLM:
    """返回给 LlamaIndex 查询引擎使用的兼容 LLM。"""

    return CompatibleLlamaIndexLLM(
        llm=_get_langchain_llm(),
        model_name=llm_model or "unknown",
        context_window=llama_index_context_window,
        num_output=llama_index_num_output,
    )


def get_embedding_model():
    return OpenAIEmbedding(
        model_name=embedding_model,
        api_key=embedding_api_key,
        api_base=embedding_api_base,
        embed_batch_size=32,
    )


def get_qdrant_client():
    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
