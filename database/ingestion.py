from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from uuid import NAMESPACE_URL, uuid5

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from config import get_embedding_model, get_qdrant_client

# 默认 docstore 文件名和存储路径
DOCSTORE_FILENAME = "docstore.json"
DEFAULT_DOCSTORE_BASE_DIR = Path(__file__).resolve().parent / "docstores"

# 默认父子分块大小和重叠
DEFAULT_PARENT_CHUNK_SIZE = 2000
DEFAULT_PARENT_CHUNK_OVERLAP = 200
DEFAULT_CHILD_CHUNK_SIZE = 500
DEFAULT_CHILD_CHUNK_OVERLAP = 100


def _make_qdrant_safe_id(raw_id: str) -> str:
    """
    把业务可读 ID 转成 Qdrant 可接受的稳定 UUID。

    原始节点 ID 例如：
    `某文件.md:parent:0:child:0`
    这种格式适合我们自己排查，但 Qdrant 的 point ID 只接受无符号整数或 UUID。

    这里用 uuid5 做确定性映射：
    - 同一个原始 ID 每次都会映射成同一个 UUID
    - 便于重复入库
    - 不影响父子节点关系
    """

    return str(uuid5(NAMESPACE_URL, raw_id))


def _load_recursive_character_text_splitter():
    """加载文本分块器，兼容不同版本的 langchain 包。"""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise ImportError(
                "RecursiveCharacterTextSplitter 不可用，请安装 `langchain-text-splitters`"
            ) from exc
    return RecursiveCharacterTextSplitter


def _build_splitter(chunk_size: int, chunk_overlap: int):
    """构建文本分块器。"""

    splitter_cls = _load_recursive_character_text_splitter()
    return splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def get_docstore_persist_path(
    collection_name: str,
    base_dir: str | Path | None = None,
) -> Path:
    """获取当前集合对应的 docstore 持久化路径。"""

    root = Path(base_dir) if base_dir is not None else DEFAULT_DOCSTORE_BASE_DIR
    return root / collection_name / DOCSTORE_FILENAME


def _read_documents(data_dir: str | Path) -> list[Document]:
    """读取目录中的所有文本文件，构造成 LlamaIndex Document。"""

    source_dir = Path(data_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {source_dir}")

    documents: list[Document] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                text=text,
                doc_id=path.name,
                metadata={
                    "file_name": path.name,
                    "source_path": str(path.resolve()),
                },
            )
        )

    if not documents:
        raise ValueError(f"数据目录中没有找到文件: {source_dir}")

    return documents


def _link_siblings(nodes: Sequence[TextNode]) -> None:
    """给一组同级节点建立前后兄弟关系。"""

    for index, node in enumerate(nodes):
        if index > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[index - 1].node_id
            )
        if index + 1 < len(nodes):
            node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[index + 1].node_id
            )


def _split_non_empty(splitter, text: str) -> list[str]:
    """只保留非空分块。"""

    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def _build_parent_child_nodes(
    documents: Iterable[Document],
    *,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> tuple[list[TextNode], list[TextNode]]:
    """
    输入文档列表，输出父节点和子节点。

    父节点用于保留较完整上下文，
    子节点用于向量化检索。
    """

    parent_splitter = _build_splitter(parent_chunk_size, parent_chunk_overlap)
    child_splitter = _build_splitter(child_chunk_size, child_chunk_overlap)

    parent_nodes: list[TextNode] = []
    child_nodes: list[TextNode] = []

    for document in documents:
        doc_id = document.doc_id or "document"
        doc_metadata = dict(document.metadata)
        doc_parent_nodes: list[TextNode] = []

        for parent_index, parent_chunk in enumerate(
            _split_non_empty(parent_splitter, document.text)
        ):
            raw_parent_id = f"{doc_id}:parent:{parent_index}"
            parent_id = _make_qdrant_safe_id(raw_parent_id)
            parent_node = TextNode(
                id_=parent_id,
                text=parent_chunk,
                metadata={
                    **doc_metadata,
                    "doc_id": doc_id,
                    "raw_node_id": raw_parent_id,
                    "chunk_level": "parent",
                    "parent_chunk_index": parent_index,
                },
            )

            sibling_child_nodes: list[TextNode] = []

            for child_index, child_chunk in enumerate(
                _split_non_empty(child_splitter, parent_chunk)
            ):
                raw_child_id = f"{raw_parent_id}:child:{child_index}"
                child_id = _make_qdrant_safe_id(raw_child_id)
                child_node = TextNode(
                    id_=child_id,
                    text=child_chunk,
                    metadata={
                        **doc_metadata,
                        "doc_id": doc_id,
                        "raw_node_id": raw_child_id,
                        "chunk_level": "child",
                        "parent_node_id": parent_id,
                        "raw_parent_node_id": raw_parent_id,
                        "parent_chunk_index": parent_index,
                        "child_chunk_index": child_index,
                    },
                )
                child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_id
                )
                sibling_child_nodes.append(child_node)

            _link_siblings(sibling_child_nodes)

            parent_node.relationships[NodeRelationship.CHILD] = [
                RelatedNodeInfo(node_id=node.node_id) for node in sibling_child_nodes
            ]

            doc_parent_nodes.append(parent_node)
            child_nodes.extend(sibling_child_nodes)

        _link_siblings(doc_parent_nodes)
        parent_nodes.extend(doc_parent_nodes)

    return parent_nodes, child_nodes


def ingest_collection(
    collection_name: str,
    data_dir: str | Path,
    *,
    client=None,
    embed_model=None,
    docstore_base_dir: str | Path | None = None,
    parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = DEFAULT_CHILD_CHUNK_OVERLAP,
) -> dict[str, str | int]:
    """
    将指定目录的文档插入到 Qdrant 向量库和本地 Docstore。

    - 父节点用于文档层级聚合
    - 子节点用于向量化检索
    """

    client = client or get_qdrant_client()
    embed_model = embed_model or get_embedding_model()

    documents = _read_documents(data_dir)
    parent_nodes, child_nodes = _build_parent_child_nodes(
        documents,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )

    # 这里仍然固定使用 1024 维，因为你当前配置的 bge-m3 / BAAI/bge-m3 就是 1024 维。
    # 如果后续更换 embedding 模型，需要同步调整这里的 size。
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"集合 {collection_name} 已创建。")
    else:
        print(f"集合 {collection_name} 已存在。")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        transformations=[embed_model],
    )
    ingestion_pipeline.run(nodes=child_nodes)

    docstore = SimpleDocumentStore()
    docstore.add_documents([*parent_nodes, *child_nodes], store_text=True)

    docstore_path = get_docstore_persist_path(collection_name, docstore_base_dir)
    docstore_path.parent.mkdir(parents=True, exist_ok=True)
    docstore.persist(persist_path=str(docstore_path))

    summary = {
        "collection_name": collection_name,
        "document_count": len(documents),
        "parent_node_count": len(parent_nodes),
        "leaf_node_count": len(child_nodes),
        "docstore_path": str(docstore_path),
    }
    print(
        "[INFO] ingestion 完成:",
        f"collection={collection_name}",
        f"documents={summary['document_count']}",
        f"parents={summary['parent_node_count']}",
        f"leaf_nodes={summary['leaf_node_count']}",
    )
    return summary


if __name__ == "__main__":
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    data_dir = Path(r"C:\Users\Geon\Downloads\policy_documents")
    collection_name = "policy_documents"
    summary = ingest_collection(
        collection_name=collection_name,
        data_dir=data_dir,
        client=client,
        embed_model=embed_model,
    )
    print("插入完成摘要:", summary)
