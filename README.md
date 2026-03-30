# Multi-Agent System 架构说明

## 1. 项目概览

本项目是一个面向政策问答场景的多智能体系统，核心目标是把封闭知识库检索、回答生成和口径审校拆分成职责清晰的阶段。

当前实现的核心设计如下：

- 工作流调度由 `CrewAI` 负责，按顺序执行 `researcher -> writer -> auditor`
- 知识检索由 `LlamaIndex + Qdrant` 负责，支持向量检索和父子分块自动合并
- 知识访问通过 `CrewAI Tool` 统一暴露给 Agent，Agent 不直接感知底层向量库细节
- 多知识库查询通过 `RouterQueryEngine` 做统一路由
- LLM 侧分为两类用途：
  - `CrewAI LLM`：驱动 Agent 执行任务
  - `LlamaIndex LLM`：驱动 Router 和检索后回答生成

---

## 2. 当前执行链路

系统入口在 `main.py`，当前实际执行链路如下：

1. 接收用户输入
2. 初始化 Qdrant 客户端
3. 为三个知识库分别构建 `RetrieverQueryEngine`
4. 把查询引擎封装成 `CrewAI Tool`
5. 创建三个 Agent：
   - `researcher`
   - `writer`
   - `auditor`
6. 创建三个顺序任务：
   - 检索与事实归纳
   - 面向用户的回答撰写
   - 口径审校与最终定稿
7. 通过 `Crew(process=Process.sequential)` 串行执行
8. 返回最终结果

对应代码主链如下：

```text
run(user_input)
  -> build_engines(client)
  -> build_toolset(engines)
  -> build_agents(toolset)
  -> build_tasks(...)
  -> Crew(..., process=Process.sequential)
  -> crew.kickoff()
```

---

## 3. 系统分层

### 3.1 协调层

协调层由 `CrewAI` 负责，主要职责：

- 管理 Agent 和 Task
- 控制执行顺序
- 传递上一步任务输出作为下一步上下文
- 输出完整执行结果

当前不是并行工作流，而是严格顺序执行。

### 3.2 Agent 层

系统包含 3 个职责分离的 Agent：

| Agent | 主要职责 | 可用工具 |
|------|------|------|
| `researcher` | 检索政策依据，区分原文与解读，输出结构化研究结果 | 原文库、解读库、统一 Router |
| `writer` | 基于研究结果撰写面向用户的回答 | 解读库、统一 Router |
| `auditor` | 审校口径、压缩风险、输出最终版本 | 审校规则库、统一 Router |

这三个 Agent 都由 `config.get_llm()` 返回的 `CrewAI LLM` 驱动。

### 3.3 检索层

每个知识库最终都会被封装成一个 `RetrieverQueryEngine`，底层结构如下：

- 向量存储：`QdrantVectorStore`
- 索引构建：`VectorStoreIndex.from_vector_store(...)`
- 基础检索：`index.as_retriever(similarity_top_k=3)`
- 可选增强：`AutoMergingRetriever`
- 对外查询接口：`RetrieverQueryEngine`

如果本地存在对应的 `docstore.json`，系统会启用 `AutoMergingRetriever`，把命中的子块自动回收到更完整的父级语义块；如果不存在，则退化为普通向量检索。

### 3.4 路由层

系统额外构建了一个统一路由工具：

- 使用 `LlamaIndex QueryEngineTool` 把多个知识库包装成候选查询源
- 使用 `RouterQueryEngine` 让 LLM 在多个知识库之间做选择
- 再通过 `tools/crewai_tools.py` 包装成 `CrewAI Tool`

这意味着 Agent 既可以直接访问专用知识库，也可以通过 Router 走统一入口。

---

## 4. 当前知识库划分

当前代码里实际使用的是三套政策知识库：

| 集合名 | 业务含义 | 在 `build_engines()` 中的键 |
|------|------|------|
| `policy_documents` | 政策原文、正式通知、制度性要求 | `technical` |
| `policy_review_rules` | 审校规则、风险提示、适用边界 | `audit` |
| `policy_interpretations` | 政策解读、背景说明、经验材料 | `shared` |

注意：

- 代码中的内部键名仍然是 `technical / audit / shared`
- 但实际集合名已经切换为政策场景下的三类知识库
- 文档、提示词和工具描述都应以政策场景理解，不应再按“技术文档库”解释

---

## 5. Tool 封装方式

Agent 并不直接调用 `LlamaIndex` 的查询引擎，而是通过 `KnowledgeQueryTool` 访问知识库。

封装分为两层：

1. `tools/router.py`
   - 构建 `QueryEngineTool`
   - 构建 `RouterQueryEngine`

2. `tools/crewai_tools.py`
   - 把 `query_engine.query(input)` 包装成 `CrewAI BaseTool`
   - 对工具名做 ASCII 安全转换
   - 保留中文业务描述，便于中文场景下的工具选择
   - 对空结果和异常结果做统一字符串化

这样做的结果是：

- Agent 层只面对统一的 CrewAI Tool 协议
- 检索层实现可以替换，而不需要改 Agent 侧接口
- Router 和普通知识库工具都能以同一种方式暴露给 Agent

---

## 6. 数据入库流程

文档入库逻辑位于 `database/ingestion.py`，当前实现不是简单的单层切块，而是父子分块结构。

### 6.1 入库步骤

1. 从目录读取原始文本文件
2. 按文件构造 `Document`
3. 先切父块，再对每个父块切子块
4. 建立父子、前后兄弟节点关系
5. 子块写入 Qdrant 向量库
6. 父块和子块一起写入本地 `SimpleDocumentStore`
7. 查询时如存在 docstore，则启用自动合并检索

### 6.2 设计目的

- 子块用于提升向量召回精度
- 父块用于保留更完整上下文
- `uuid5` 用于把业务节点 ID 稳定映射成 Qdrant 可接受的 UUID
- `docstore.json` 用于支持 `AutoMergingRetriever`

### 6.3 入库示意

```text
原始文件
  -> Document
  -> 父块切分
  -> 子块切分
  -> 子块写入 Qdrant
  -> 父块 + 子块写入本地 Docstore
  -> 查询时按需自动合并回父块
```

---

## 7. 配置与模型适配

配置集中在 `config.py`。

### 7.1 LLM 配置

项目区分两类模型入口：

- `get_llm()`
  - 返回 `CrewAI LLM`
  - 用于 `researcher / writer / auditor`

- `get_llama_index_llm()`
  - 返回 `CompatibleLlamaIndexLLM`
  - 底层通过 `langchain_openai.ChatOpenAI` 适配
  - 用于 `RetrieverQueryEngine` 和 `RouterQueryEngine`

这样设计的原因是：

- CrewAI 直接需要自己的 LLM 对象
- LlamaIndex 侧需要一个兼容第三方 OpenAI 接口的 LLM 包装层
- 项目希望兼容自定义模型名和 OpenAI 兼容网关

### 7.2 Embedding 与向量库配置

- Embedding 通过 `OpenAIEmbedding` 初始化
- 默认 embedding 模型为 `bge-m3`
- Qdrant 通过 `QdrantClient(url, api_key)` 连接

当前入库时向量维度固定为 `1024`，如果更换 embedding 模型，需要同步调整集合向量维度。

---

## 8. 数据流说明

### 8.1 研究阶段

- `researcher` 接收用户问题
- 调用政策原文库、解读库或 Router
- 提炼政策依据、适用对象、条件、时间范围、限制和不确定项
- 输出结构化研究结果

### 8.2 写作阶段

- `writer` 读取研究结果
- 生成面向用户的中文答复
- 明确区分政策原文依据与解释性内容
- 对资料不足的部分保留不确定性表达

### 8.3 审校阶段

- `auditor` 读取写作结果
- 结合审校规则库检查过度推断、条件遗漏和口径不稳
- 必要时直接改写
- 输出最终版本

---

## 9. 项目目录结构

当前仓库的主要结构如下：

```text
multi_agent_system/
├── main.py
├── config.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── agents/
│   ├── factory.py
│   ├── researcher.py
│   ├── writer.py
│   └── auditor.py
├── tools/
│   ├── knowledge_tools.py
│   ├── crewai_tools.py
│   └── router.py
├── workflows/
│   └── pipeline.py
└── database/
    └── ingestion.py
```

补充说明：

- `tools/knowledge_tools.py` 负责构建 retriever 和 query engine
- `tools/crewai_tools.py` 负责把 query engine 适配成 CrewAI Tool
- `agents/factory.py` 负责按职责给不同 Agent 分配工具
- `workflows/pipeline.py` 定义三阶段 Task
- `database/docstores/` 会在入库后按集合名持久化本地 docstore，但该目录不一定预先存在于仓库

---

## 10. 当前架构特点

### 优点

- 职责边界清晰，检索、写作、审校分阶段执行
- 知识访问统一通过 Tool 暴露，便于替换底层实现
- 父子分块 + AutoMergingRetriever 能在精度和上下文完整性之间做平衡
- Router 提供跨知识库统一入口，适合问题边界不明确的场景

### 当前限制

- 当前流程是严格串行，不支持并行协作
- Router 和回答生成都依赖 LLM，因此结果存在一定波动
- `similarity_top_k` 当前固定为 `3`，召回覆盖面偏保守
- 集合内部键名 `technical / audit / shared` 与真实业务语义不完全一致，只是历史命名保留

---

## 11. 总结

当前系统已经从泛化的“技术文档问答”形态，收敛为面向政策问答的三阶段多智能体架构：

- `researcher` 负责找依据
- `writer` 负责写答复
- `auditor` 负责压口径
- `Qdrant + LlamaIndex` 负责封闭知识检索
- `Router + CrewAI Tool` 负责统一知识访问协议

如果后续继续演进，最可能的方向包括：

- 增加检索日志和命中证据输出
- 固化 Router 选择策略以提高稳定性
- 增加 rerank、metadata filter 或引用追踪能力
- 把内部键名从 `technical / audit / shared` 进一步收敛为与政策场景一致的命名
