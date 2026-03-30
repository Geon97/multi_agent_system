from crewai import Crew, Process

from agents.factory import build_agents
from config import get_qdrant_client
from tools.crewai_tools import build_crewai_tool
from tools.knowledge_tools import build_engine
from tools.router import build_router_engine, build_router_source
from workflows.pipeline import build_tasks

# 这三个集合分别承载：
# 1. 政策原文与正式文件
# 2. 口径审校规则、适用边界和风险提示
# 3. 政策解读、背景说明和补充性材料
# 这里仍然使用 ASCII 集合名，避免与向量库、文件目录或外部工具的兼容性冲突。
POLICY_DOCUMENTS_COLLECTION = "policy_documents"
POLICY_REVIEW_RULES_COLLECTION = "policy_review_rules"
POLICY_INTERPRETATIONS_COLLECTION = "policy_interpretations"

# 工具名称和说明直接暴露给 Agent。
# 名称尽量使用中文，帮助模型在中文场景下更自然地选择工具。
POLICY_DOCUMENTS_TOOL_NAME = "政策原文库"
POLICY_DOCUMENTS_TOOL_DESCRIPTION = (
    "用于查询政策原文、规范条文、正式通知和制度性要求。"
    "当问题涉及政策依据、适用条款、执行口径或政策硬性要求时，优先使用该工具。"
)

POLICY_REVIEW_RULES_TOOL_NAME = "政策审校规则库"
POLICY_REVIEW_RULES_TOOL_DESCRIPTION = (
    "用于查询政策适用边界、审核关注点、风险提示、常见限制条件和合规要求。"
    "当需要核对口径、识别风险或审查答复是否稳妥时，优先使用该工具。"
)

POLICY_INTERPRETATIONS_TOOL_NAME = "政策解读库"
POLICY_INTERPRETATIONS_TOOL_DESCRIPTION = (
    "用于查询政策解读、背景说明、办事经验、概念解释和补充性知识。"
    "当需要帮助理解政策含义、适用场景或形成更易读的中文解释时，优先使用该工具。"
)

UNIFIED_ROUTER_TOOL_NAME = "政策知识路由器"
UNIFIED_ROUTER_TOOL_DESCRIPTION = (
    "用于跨多个政策知识库进行综合检索。"
    "当你不确定应当查询政策原文、政策解读还是审校规则时，使用该工具进行统一路由。"
)


def build_engines(client):
    """构建三个面向政策场景的 LlamaIndex 查询引擎。"""

    return {
        "technical": build_engine(client, POLICY_DOCUMENTS_COLLECTION),
        "audit": build_engine(client, POLICY_REVIEW_RULES_COLLECTION),
        "shared": build_engine(client, POLICY_INTERPRETATIONS_COLLECTION),
    }


def build_toolset(engines):
    """把 LlamaIndex 引擎包装成 Agent 可直接调用的 CrewAI 工具。"""

    # Router 仍然依赖 LlamaIndex 自己的 QueryEngineTool 作为候选源，
    # 因此这里先构造路由源，再把最终路由引擎包装成 CrewAI 工具。
    router_sources = [
        build_router_source(
            engines["technical"],
            POLICY_DOCUMENTS_TOOL_NAME,
            POLICY_DOCUMENTS_TOOL_DESCRIPTION,
        ),
        build_router_source(
            engines["audit"],
            POLICY_REVIEW_RULES_TOOL_NAME,
            POLICY_REVIEW_RULES_TOOL_DESCRIPTION,
        ),
        build_router_source(
            engines["shared"],
            POLICY_INTERPRETATIONS_TOOL_NAME,
            POLICY_INTERPRETATIONS_TOOL_DESCRIPTION,
        ),
    ]
    router_engine = build_router_engine(router_sources)

    return {
        "technical": build_crewai_tool(
            engines["technical"],
            POLICY_DOCUMENTS_TOOL_NAME,
            POLICY_DOCUMENTS_TOOL_DESCRIPTION,
        ),
        "audit": build_crewai_tool(
            engines["audit"],
            POLICY_REVIEW_RULES_TOOL_NAME,
            POLICY_REVIEW_RULES_TOOL_DESCRIPTION,
        ),
        "shared": build_crewai_tool(
            engines["shared"],
            POLICY_INTERPRETATIONS_TOOL_NAME,
            POLICY_INTERPRETATIONS_TOOL_DESCRIPTION,
        ),
        "router": build_crewai_tool(
            router_engine,
            UNIFIED_ROUTER_TOOL_NAME,
            UNIFIED_ROUTER_TOOL_DESCRIPTION,
        ),
    }


def run(user_input):
    """
    系统统一入口。

    保持 `run(user_input)` 不变，方便后续命令行调用、脚本调用或上层接口接入。
    """

    client = get_qdrant_client()
    engines = build_engines(client)
    toolset = build_toolset(engines)
    agents = build_agents(toolset)
    tasks = build_tasks(
        user_input,
        agents["researcher"],
        agents["writer"],
        agents["auditor"],
    )

    crew = Crew(
        agents=[
            agents["researcher"],
            agents["writer"],
            agents["auditor"],
        ],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    return crew.kickoff()


if __name__ == "__main__":
    # 知识库研究助手
    user_input = "对比古城区拆迁补贴政策和安置政策的差异。"
    output = run(user_input)
    print("\n================ 最终结果 ================\n")
    print(output)
