# Multi-Agent System 架构说明

## 1. 项目概览

本项目是一个多智能体系统 (Multi-Agent System)，核心设计如下：

- 顶层工作流调度：CrewAI 负责任务分配和执行顺序控制
- Agent 内部执行逻辑：每个 Agent 使用 LangChain 执行具体任务
- 知识库支持：LlamaIndex + Qdrant 向量库，封闭系统，仅提供内部数据查询
- 智能路由：Router 工具自动选择最适合的知识库
- 输出层：综合分析，标准化输出，推理日志可审计

---

## 2. 架构层级

    用户输入
       │
       ▼
    CrewAI 工作流协调层
       ├─ 分配任务给 Agent
       ├─ 顺序或并行控制
       └─ 任务状态追踪 & 日志
           │
           ▼
    Agent 层（LangChain 内部执行）
       ├─ Researcher Agent
       │   ├─ 查询 LlamaIndex 技术文档库
       │   ├─ 生成结构化技术分析
       │   └─ 输出推理日志
       │
       ├─ Writer Agent
       │   ├─ 查询 LlamaIndex 共享知识库
       │   ├─ 整理研究结果为完整文档
       │   └─ 输出推理日志
       │
       └─ Auditor Agent
           ├─ 查询 LlamaIndex 审计规则库
           ├─ 校验 Writer 输出
           └─ 输出优化后的最终报告
               │
               ▼
    共享知识索引层（Qdrant + LlamaIndex）
       ├─ technical_docs
       ├─ audit_rules
       └─ shared_knowledge
               │
               ▼
    综合分析 / 最终输出
       ├─ 冲突合并
       ├─ 标准化输出
       ├─ 可审计推理日志
       └─ 返回用户

---

## 3. 数据流说明

1. 用户输入 → CrewAI 协调层
   - CrewAI 根据工作流顺序分配任务给各 Agent
   - 支持顺序执行 (Process.sequential) 或并行执行

2. Agent 内部处理
   - LangChain + QueryEngine 查询 LlamaIndex 向量库
   - 各 Agent 有独立目标：Researcher 查询技术、Writer 整理文档、Auditor 审核
   - 输出推理日志，供综合分析层使用

3. 知识库管理
   - 向量库使用 Qdrant 存储，LlamaIndex 构建查询引擎
   - 文档通过 RecursiveCharacterTextSplitter 分块
   - 插入流程使用 add_documents，可记录父文档信息用于追溯

4. Router 工具
   - 根据问题智能选择最相关的知识库
   - 所有 Agent 内部共享 Router，统一知识查询入口

5. 综合分析 & 输出
   - 汇总 Agent 输出
   - 冲突合并、标准化格式
   - 输出最终可审计报告

---

## 4. 知识库插入流程示意

    原始文档
       │
       ▼
    RecursiveCharacterTextSplitter
       └─ 文本分块
           │
           ▼
    Document + TextNode
       └─ 保存父子/来源信息
           │
           ▼
    LlamaIndex VectorStoreIndex
       └─ Qdrant 向量库

- 插入完成后，所有 Agent 查询时都可通过 Router 获取向量化内容

---

## 5. Agent 角色和目标

| Agent        | 目标 | 功能 |
|--------------|------|------|
| Researcher   | 获取技术信息 | 分析问题核心，查询技术向量库，生成结构化技术分析 |
| Writer       | 生成文档 | 基于 Researcher 输出整理为高质量技术方案 |
| Auditor      | 审核和优化 | 校验 Writer 输出，检查事实和逻辑，应用审计规则库 |
| Router Tool  | 知识路由 | 智能选择最相关知识库，统一查询接口 |

---

## 6. 核心工具和依赖

- CrewAI：工作流调度、任务分配、状态追踪
- LangChain：Agent 内部逻辑执行
- LlamaIndex：构建向量索引，支持 QueryEngine
- Qdrant：存储向量化文档，支持高性能向量检索
- RecursiveCharacterTextSplitter：文档分块
- RouterQueryEngine：多知识库智能路由

---

## 7. 项目目录结构

    multi_agent_system/
    │
    ├── main.py                 # 入口，执行 CrewAI 工作流
    ├── config.py               # LLM 配置、Qdrant 客户端
    │
    ├── agents/
    │   ├── researcher.py
    │   ├── writer.py
    │   └── auditor.py
    │
    ├── tools/
    │   ├── knowledge_tools.py  # 各知识库 Tool 封装
    │   └── router.py           # Router 工具
    │
    ├── workflows/
    │   └── pipeline.py         # Task 流程定义
    │
    └── database/               # Qdrant 数据集
        └── ingestion.py        # 文档分块方法

---

## 8. 总结

- 稳定执行：CrewAI 控制工作流顺序，保证依赖任务严格执行
- 可扩展：增加 Agent 或知识库无需修改核心协调逻辑
- 高性能查询：Qdrant + LlamaIndex 提供向量化检索
- 可追溯和审计：每个 Agent 输出推理日志，综合分析层标准化输出
- 封闭系统：只读取向量库，不访问外部网络，保证数据安全