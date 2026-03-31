from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeRelationship, NodeWithScore, RelatedNodeInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import get_embedding_model, get_llama_index_llm


def _normalize_identity(value) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text == "None":
        return None
    return text


def _resolve_node_doc_id(node) -> str | None:
    if _normalize_identity(node.ref_doc_id):
        return _normalize_identity(node.ref_doc_id)

    metadata = node.metadata or {}
    for key in ("doc_id", "document_id", "ref_doc_id", "file_name"):
        value = _normalize_identity(metadata.get(key))
        if value:
            return value

    source_path = _normalize_identity(metadata.get("source_path"))
    if source_path:
        return Path(source_path).name

    return None


def _hydrate_node_identity(node):
    """兼容旧数据：没有 docstore 时，尽量从 metadata 还原文档身份。"""

    doc_id = _resolve_node_doc_id(node)
    if not doc_id:
        return node

    node.metadata["file_name"] = _normalize_identity(node.metadata.get("file_name")) or doc_id
    node.metadata["document_id"] = doc_id
    node.metadata["doc_id"] = doc_id
    node.metadata["ref_doc_id"] = doc_id

    if node.source_node is None:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc_id,
            metadata={"file_name": node.metadata["file_name"]},
        )

    return node


def _merge_key(node) -> tuple[str, str, int] | None:
    chunk_level = _normalize_identity(node.metadata.get("chunk_level"))
    parent_node_id = _normalize_identity(node.metadata.get("parent_node_id"))
    doc_id = _resolve_node_doc_id(node)
    parent_chunk_index = node.metadata.get("parent_chunk_index")

    if (
        chunk_level != "child"
        or not parent_node_id
        or not doc_id
        or not isinstance(parent_chunk_index, int)
    ):
        return None

    return doc_id, parent_node_id, parent_chunk_index


def _merge_group(items: list[NodeWithScore]) -> NodeWithScore:
    unique_items: list[NodeWithScore] = []
    seen_child_indexes: set[int | None] = set()

    for item in sorted(
        items,
        key=lambda current: (
            current.node.metadata.get("child_chunk_index", 0),
            -(current.score or 0.0),
        ),
    ):
        child_index = item.node.metadata.get("child_chunk_index")
        if child_index in seen_child_indexes:
            continue
        seen_child_indexes.add(child_index)
        unique_items.append(item)

    best_item = max(unique_items, key=lambda current: current.score or 0.0)
    merged_node = best_item.node.model_copy(deep=True)
    merged_node.text = "\n\n".join(
        text
        for text in (item.node.text.strip() for item in unique_items)
        if text
    )
    merged_node.metadata["matched_child_chunk_indexes"] = [
        item.node.metadata.get("child_chunk_index") for item in unique_items
    ]
    merged_node.metadata["matched_chunk_count"] = len(unique_items)
    _hydrate_node_identity(merged_node)

    return NodeWithScore(node=merged_node, score=best_item.score)


def _postprocess_retrieved_nodes(
    nodes: Iterable[NodeWithScore],
    *,
    final_top_k: int,
) -> list[NodeWithScore]:
    """检索后统一补全文档身份，并合并同父块的相邻命中。"""

    hydrated_nodes: list[NodeWithScore] = []
    for item in nodes:
        cloned_node = item.node.model_copy(deep=True)
        _hydrate_node_identity(cloned_node)
        if not cloned_node.text.strip():
            continue
        hydrated_nodes.append(NodeWithScore(node=cloned_node, score=item.score))

    grouped_nodes: dict[tuple | str, list[NodeWithScore]] = defaultdict(list)
    for item in hydrated_nodes:
        key = _merge_key(item.node)
        if key is None:
            key = f"node:{item.node.node_id}"
        grouped_nodes[key].append(item)

    merged_nodes = [_merge_group(group) for group in grouped_nodes.values()]
    merged_nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
    return merged_nodes[:final_top_k]


class PostprocessedRetriever(BaseRetriever):
    """对底层向量检索结果做补全、去重和轻量合并。"""

    def __init__(self, base_retriever: BaseRetriever, *, final_top_k: int) -> None:
        super().__init__(callback_manager=base_retriever.callback_manager)
        self._base_retriever = base_retriever
        self._final_top_k = final_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = self._base_retriever.retrieve(query_bundle)
        return _postprocess_retrieved_nodes(nodes, final_top_k=self._final_top_k)


def build_retriever(
    client,
    collection_name,
    *,
    embed_model=None,
    similarity_top_k: int = 3,
):
    """
    直接返回 Qdrant 检索器，不依赖本地 docstore。

    优化点：
    1. 底层多取一些候选，给去重/合并留空间；
    2. 检索后补全文档身份字段；
    3. 命中同一父块的多个子块时自动轻量合并。
    """

    embed_model = embed_model or get_embedding_model()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    raw_similarity_top_k = max(similarity_top_k * 2, similarity_top_k + 2)
    vector_retriever = index.as_retriever(similarity_top_k=raw_similarity_top_k)
    return PostprocessedRetriever(
        vector_retriever,
        final_top_k=similarity_top_k,
    )


def build_engine(
    client,
    collection_name: str,
    *,
    embed_model=None,
    llm=None,
    similarity_top_k: int = 3,
):
    """
    基于 Qdrant 直接构建 LlamaIndex 查询引擎。
    """

    retriever = build_retriever(
        client,
        collection_name,
        embed_model=embed_model,
        similarity_top_k=similarity_top_k,
    )
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm or get_llama_index_llm(),
    )
