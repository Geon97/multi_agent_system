from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeCollection:
    key: str
    name: str
    description: str

# 定义现有知识库集合名和中文说明，作为统一路由的候选源
KNOWLEDGE_COLLECTIONS: tuple[KnowledgeCollection, ...] = (
    KnowledgeCollection(
        key="policy_documents",
        name="政策原文库",
        description="政策原文、正式通知、制度性要求。",
    ),
    KnowledgeCollection(
        key="policy_review_rules",
        name="政策审核规则库",
        description="审核关注点、适用边界、风险提示和限制条件。",
    ),
    KnowledgeCollection(
        key="policy_interpretations",
        name="政策解读库",
        description="政策解读、背景说明、经验知识和补充材料。",
    ),
)

DEFAULT_PER_COLLECTION_TOP_K = 3
DEFAULT_FINAL_TOP_K = 8
