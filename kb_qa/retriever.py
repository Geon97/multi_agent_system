from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from config import get_embedding_model, get_qdrant_client
from kb_qa.constants import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_PER_COLLECTION_TOP_K,
    KNOWLEDGE_COLLECTIONS,
    KnowledgeCollection,
)
from tools.knowledge_tools import build_retriever


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    collection_key: str
    collection_name: str
    collection_description: str
    file_name: str | None
    source_path: str | None
    score: float | None
    weighted_score: float
    evidence_type: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DATE_PATTERN = re.compile(
    r"(?:\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d{4}-\d{1,2}-\d{1,2}|截至\d{4}年\d{1,2}月)"
)
TIME_QUESTION_KEYWORDS = ("什么时候", "何时", "时间", "日期", "完成时间", "整改完成")
TIME_DIRECT_KEYWORDS = ("完成", "整改完成", "完工", "竣工", "结束", "验收", "完成时限", "完成日期")
TIME_INDIRECT_KEYWORDS = ("规划", "编制", "实施", "推进", "期限", "时限", "按照规划", "保护整治")
COLLECTION_WEIGHT_BONUS = {
    "policy_documents": 0.35,
    "policy_interpretations": 0.15,
    "policy_review_rules": -0.05,
}


def _is_time_question(question: str) -> bool:
    return any(keyword in question for keyword in TIME_QUESTION_KEYWORDS)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _classify_evidence(question: str, text: str) -> str:
    if not _is_time_question(question):
        return "context"

    if DATE_PATTERN.search(text) and _contains_any(text, TIME_DIRECT_KEYWORDS):
        return "direct"

    if _contains_any(text, TIME_DIRECT_KEYWORDS) or _contains_any(text, TIME_INDIRECT_KEYWORDS):
        return "indirect"

    return "background"


def _keyword_overlap_bonus(question: str, text: str) -> float:
    keywords = [part for part in re.split(r"\s+", question) if part]
    if len(question) <= 12:
        keywords.append(question)

    matches = sum(1 for keyword in keywords if keyword and keyword in text)
    return min(matches * 0.03, 0.18)


def _rank_chunk(question: str, collection_key: str, score: float | None, text: str, evidence_type: str) -> float:
    weighted_score = score or 0.0
    weighted_score += COLLECTION_WEIGHT_BONUS.get(collection_key, 0.0)
    weighted_score += _keyword_overlap_bonus(question, text)

    if _is_time_question(question):
        if evidence_type == "direct":
            weighted_score += 0.45
        elif evidence_type == "indirect":
            weighted_score += 0.2
        else:
            weighted_score -= 0.05

        if DATE_PATTERN.search(text):
            weighted_score += 0.08

    return weighted_score

# 复用现有 tools/knowledge_tools.py 的能力，构建多库统一检索
class MultiCollectionRetriever:
    def __init__(
        self,
        *,
        client=None,
        embed_model=None,
        per_collection_top_k: int = DEFAULT_PER_COLLECTION_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        self._client = client or get_qdrant_client()
        self._embed_model = embed_model or get_embedding_model()
        self._per_collection_top_k = per_collection_top_k
        self._final_top_k = final_top_k
        self._retrievers = self._build_retrievers()

    def _build_retrievers(self) -> list[tuple[KnowledgeCollection, Any]]:
        retrievers: list[tuple[KnowledgeCollection, Any]] = []

        for collection in KNOWLEDGE_COLLECTIONS:
            if not self._client.collection_exists(collection.key):
                continue

            retriever = build_retriever(
                self._client,
                collection.key,
                embed_model=self._embed_model,
                similarity_top_k=self._get_collection_top_k(collection.key),
            )
            retrievers.append((collection, retriever))

        if not retrievers:
            raise ValueError("未找到可用知识库集合，请先确认 Qdrant 中已有现有知识库数据。")

        return retrievers

    def _get_collection_top_k(self, collection_key: str) -> int:
        if collection_key == "policy_documents":
            return self._per_collection_top_k + 1
        return self._per_collection_top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        query = question.strip()
        if not query:
            raise ValueError("问题不能为空。")

        chunks: list[RetrievedChunk] = []
        for collection, retriever in self._retrievers:
            for item in retriever.retrieve(query):
                text = item.node.text.strip()
                if not text:
                    continue
                evidence_type = _classify_evidence(query, text)
                weighted_score = _rank_chunk(
                    query,
                    collection.key,
                    item.score,
                    text,
                    evidence_type,
                )
                chunks.append(
                    RetrievedChunk(
                        collection_key=collection.key,
                        collection_name=collection.name,
                        collection_description=collection.description,
                        file_name=item.node.metadata.get("file_name"),
                        source_path=item.node.metadata.get("source_path"),
                        score=item.score,
                        weighted_score=weighted_score,
                        evidence_type=evidence_type,
                        text=text,
                    )
                )

        deduped_chunks: dict[tuple[str | None, str], RetrievedChunk] = {}
        for chunk in chunks:
            dedupe_key = (chunk.file_name, chunk.text)
            current = deduped_chunks.get(dedupe_key)
            if current is None or chunk.weighted_score > current.weighted_score:
                deduped_chunks[dedupe_key] = chunk

        ranked_chunks = sorted(
            deduped_chunks.values(),
            key=lambda current: (current.weighted_score, current.score or 0.0),
            reverse=True,
        )
        return ranked_chunks[: self._final_top_k]


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "未检索到可用知识片段。"

    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[{index}] 知识库: {chunk.collection_name} ({chunk.collection_key})",
                    f"[{index}] 说明: {chunk.collection_description}",
                    f"[{index}] 文件: {chunk.file_name or '未知文件'}",
                    f"[{index}] 分数: {chunk.score if chunk.score is not None else 'unknown'}",
                    f"[{index}] 重排分数: {chunk.weighted_score:.4f}",
                    f"[{index}] 证据类型: {chunk.evidence_type}",
                    f"[{index}] 内容:",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(sections)
