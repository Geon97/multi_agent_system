from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import get_embedding_model, get_llama_index_llm
from database.ingestion import get_docstore_persist_path


def build_retriever(
    client,
    collection_name,
    *,
    embed_model=None,
    docstore_base_dir: str | Path | None = None,
    similarity_top_k: int = 3,
    auto_merge_ratio: float = 0.5,
):
    """
    构建底层检索器。

    这里先从 Qdrant 读取向量索引，再根据 docstore 是否存在，
    决定是否启用 AutoMergingRetriever。

    这么做的原因是：
    - 如果只有向量索引，没有持久化 docstore，就退化为普通向量检索。
    - 如果 docstore 存在，就可以把切碎的子节点自动合并回更完整的父级语义块，
      对政策文件这类长文更有帮助。
    """

    embed_model = embed_model or get_embedding_model()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    docstore_path = get_docstore_persist_path(collection_name, docstore_base_dir)
    if not docstore_path.exists():
        # 没有 docstore 时，直接返回普通向量检索器。
        return vector_retriever

    docstore = SimpleDocumentStore.from_persist_path(str(docstore_path))
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
    )
    return AutoMergingRetriever(
        vector_retriever,
        storage_context,
        simple_ratio_thresh=auto_merge_ratio,
    )


def build_engine(
    client,
    collection_name,
    *,
    embed_model=None,
    llm=None,
    docstore_base_dir: str | Path | None = None,
    similarity_top_k: int = 3,
    auto_merge_ratio: float = 0.5,
):
    """
    基于 retriever 构建最终的 LlamaIndex 查询引擎。

    这一层仍然只负责“检索 + 回答生成”，不直接暴露给 Agent。
    Agent 侧看到的工具协议由 `tools/crewai_tools.py` 统一适配。
    """

    retriever = build_retriever(
        client,
        collection_name,
        embed_model=embed_model,
        docstore_base_dir=docstore_base_dir,
        similarity_top_k=similarity_top_k,
        auto_merge_ratio=auto_merge_ratio,
    )
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm or get_llama_index_llm(),
    )
