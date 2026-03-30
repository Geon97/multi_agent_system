from agents.auditor import create_auditor
from agents.researcher import create_researcher
from agents.writer import create_writer


def build_agents(toolset):
    """
    按职责边界为不同 Agent 分配工具。

    这里采用“专用工具优先，统一路由兜底”的策略：
    - researcher 优先查政策原文和政策解读
    - writer 优先查政策解读
    - auditor 优先查审校规则
    - router 作为跨库问题的兜底入口
    """

    researcher_tools = [
        toolset["technical"],
        toolset["shared"],
        toolset["router"],
    ]
    writer_tools = [
        toolset["shared"],
        toolset["router"],
    ]
    auditor_tools = [
        toolset["audit"],
        toolset["router"],
    ]

    return {
        "researcher": create_researcher(researcher_tools),
        "writer": create_writer(writer_tools),
        "auditor": create_auditor(auditor_tools),
    }
