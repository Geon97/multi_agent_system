# Tool Adapter And Agent Boundary Design

## Context

The current project direction is sound:

- `CrewAI` coordinates multi-agent workflow execution.
- `LlamaIndex` owns retrieval, routing, and vector store access.
- `Qdrant` stores vectors.

The main structural issue is not the high-level architecture. It is that the tool protocol and role boundaries are still loose:

- Agents currently receive `llama_index.core.tools.QueryEngineTool` objects directly.
- All three agents depend on the same router-first entrypoint.
- `main.py` mixes engine creation, tool creation, agent wiring, and execution.

This design tightens those boundaries without changing the public runtime entrypoint: `run(user_input)`.

## Goals

- Keep `LlamaIndex` as the retrieval implementation.
- Expose only `CrewAI`-native tools to agents.
- Make agent roles depend on explicit tool boundaries, not only prompts.
- Reduce assembly logic inside `main.py`.
- Preserve current external behavior and `run(user_input)` signature.

## Non-Goals

- No change to retrieval quality strategy in this pass.
- No change to task sequencing behavior.
- No change to vector store schema or ingestion pipeline.
- No change to the public run API.

## Recommended Approach

Use a thin adapter layer:

- `LlamaIndex` continues to build query engines and router engines.
- A new `CrewAI` tool adapter wraps each engine behind a uniform `BaseTool`.
- Agents consume only the adapted `CrewAI` tools.
- Router becomes a fallback tool, not the only tool for every agent.

This avoids the unnecessary bridge chain of `LlamaIndex -> LangChain -> CrewAI`.

## Module Responsibilities

### `tools/knowledge_tools.py`

Owns retrieval construction only:

- `build_retriever(...)`
- `build_engine(...)`

It must not return agent-facing tool objects.

### `tools/router.py`

Owns router engine construction only:

- build a `RouterQueryEngine`
- accept the source `QueryEngineTool` objects required by `LlamaIndex` router selection

It must not return agent-facing tool objects.

### `tools/crewai_tools.py`

New module. Owns the adaptation boundary from retrieval engine to agent tool.

Responsibilities:

- define a `KnowledgeQueryTool` based on `CrewAI BaseTool`
- hold the target query engine
- expose a single input field: `input`
- return normalized string output
- convert retrieval failures into stable, readable tool errors

Public helper:

- `build_crewai_tool(query_engine, name, description)`

### `agents/factory.py`

New module. Owns agent construction and tool assignment.

Responsibilities:

- create `Researcher`, `Writer`, and `Auditor`
- bind each role to a role-specific tool set
- keep LLM configuration consistent

### `main.py`

Becomes composition-only code.

Responsibilities:

- get infrastructure clients
- build the three knowledge engines
- build the router engine
- adapt engines into `CrewAI` tools
- create agents via the factory
- create tasks and run the crew

It should not hold role-specific tool assignment rules inline.

## Tool Contract

All agent-visible tools should share one contract:

- input schema: one required string field named `input`
- output: plain string
- failure mode: human-readable error string that preserves enough context for retry or fallback

Recommended behavior inside the adapter:

- call `query_engine.query(input)`
- stringify the response with `str(response)`
- optionally include a short prefix on failures such as `Knowledge query failed: ...`

This gives a stable protocol even if the retrieval implementation changes later.

## Agent Tool Assignment

### Researcher

Primary tools:

- `technical_knowledge_base`
- `shared_knowledge_base`

Fallback:

- `unified_knowledge_router`

### Writer

Primary tools:

- `shared_knowledge_base`

Fallback:

- `unified_knowledge_router`

### Auditor

Primary tools:

- `audit_rules_base`

Fallback:

- `unified_knowledge_router`

## Data Flow

1. `main.py` builds the three knowledge query engines.
2. `main.py` creates the `LlamaIndex` router engine from the three source engines.
3. `main.py` adapts each engine into a `CrewAI` tool.
4. `agents/factory.py` creates agents with role-scoped tool sets.
5. `workflows/pipeline.py` runs the same task sequence as before.
6. `CrewAI` interacts only with adapted tools; retrieval internals stay behind the adapter.

## Error Handling

The adapter layer should normalize the most common problems:

- missing collection or docstore state
- query engine runtime exceptions
- provider or connectivity errors returned through the retrieval call

The tool should return a readable failure string instead of leaking raw tracebacks into the agent loop.

## Testing Strategy

This change should be verified at three levels:

- unit test the adapter with a fake query engine
- integration test `main.run(...)` with mocked engines or a lightweight fixture
- manual smoke test that each role can use its intended tool set and still reach the router fallback

At minimum, the refactor should prove:

- the adapter passes the query text correctly
- tool output is normalized to string
- each agent receives the expected tools
- `run(user_input)` still executes with the same signature

## Migration Steps

1. Remove agent-facing tool construction from `tools/knowledge_tools.py`.
2. Change `tools/router.py` to return a router engine instead of a router tool.
3. Add `tools/crewai_tools.py`.
4. Add `agents/factory.py`.
5. Update `main.py` to compose engines, adapted tools, and agents.
6. Verify `run(user_input)` remains unchanged.

## Risks

- `CrewAI BaseTool` input schema must match the agent executor's expectations.
- Router behavior may shift slightly because specialized tools are no longer hidden behind a router-only interface.
- Existing prompt text may assume a single shared tool and may need a follow-up prompt tuning pass.

## Decision

Proceed with:

- `LlamaIndex` for retrieval and routing
- `CrewAI`-native adapted tools for agent consumption
- role-scoped tool assignment with router fallback
- composition-only `main.py`
