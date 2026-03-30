import hashlib
from typing import Any

from crewai.tools.base_tool import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name


def _build_protocol_tool_name(display_name: str) -> str:
    """
    把面向业务的中文工具名转换成协议层可用的工具名。

    设计原则：
    1. 优先复用 CrewAI 自带的清洗逻辑
    2. 如果原名称是纯中文，清洗后可能变成空字符串，此时自动回退到稳定的 ASCII 名称
    3. 协议名只服务于 function calling，不承担对用户展示的职责
    """

    sanitized_name = sanitize_tool_name(display_name)
    if sanitized_name:
        return sanitized_name

    short_hash = hashlib.sha256(display_name.encode("utf-8")).hexdigest()[:12]
    return f"knowledge_tool_{short_hash}"


def _build_tool_description(display_name: str, description: str) -> str:
    """
    在描述中保留中文业务名，帮助模型在中文场景下理解工具用途。

    注意这里是“给模型看的说明”，而不是底层协议名。
    """

    return f"工具中文名：{display_name}\n{description}"


class KnowledgeQueryTool(BaseTool):
    """
    把 LlamaIndex 查询引擎适配成 CrewAI 可直接消费的工具。

    这样 Agent 层只需要面对统一的 CrewAI 工具协议，
    不必感知底层是 Qdrant、LlamaIndex 还是别的检索实现。
    """

    query_engine: Any

    @staticmethod
    def _normalize_query_output(value: Any) -> str:
        """
        把检索结果标准化成非空字符串。

        CrewAI 在原生工具调用链路里，如果工具执行结果是空串，
        会把这次 LLM 结果判定为无效并直接抛错。
        因此这里需要把“无结果”显式表达出来，而不是返回空值。
        """

        if value is None:
            return "未检索到可直接引用的政策内容。"

        text = str(value).strip()
        if not text:
            return "未检索到可直接引用的政策内容。"

        return text

    def _run(self, input: str) -> str:
        """
        执行一次自然语言查询。

        这里统一只接受一个 `input` 参数，避免不同工具的参数风格不一致。
        返回值也统一成字符串，方便 Agent 在后续推理中直接拼接和引用。
        """

        try:
            response = self.query_engine.query(input)
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            return f"Knowledge query failed: {detail}"

        return self._normalize_query_output(response)


def build_crewai_tool(query_engine: Any, name: str, description: str) -> KnowledgeQueryTool:
    """
    构建一个面向 Agent 的知识查询工具。

    这里显式区分两层命名：
    - `name` 仍然使用业务语义上的中文名，便于维护和理解
    - 真正传给 OpenAI function calling 的工具名，会转成合法的 ASCII 协议名
    """

    return KnowledgeQueryTool(
        name=_build_protocol_tool_name(name),
        description=_build_tool_description(name, description),
        query_engine=query_engine,
    )
