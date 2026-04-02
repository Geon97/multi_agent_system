from __future__ import annotations

from collections import Counter
import re
from typing import Any

from kb_qa.chains import build_answer_chain
from kb_qa.retriever import DATE_PATTERN, MultiCollectionRetriever, format_context

DETAILED_DATE_PATTERN = re.compile(
    r"(?:\d{4}年\d{1,2}月(?:\d{1,2}日|底|月底|末|上旬|中旬|下旬)?|\d{4}-\d{1,2}-\d{1,2})"
)

# 核心接口
class KnowledgeBaseQAService:
    def __init__(self, *, retriever=None, answer_chain=None) -> None:
        self._retriever = retriever or MultiCollectionRetriever()
        self._answer_chain = answer_chain or build_answer_chain()

    def ask(self, question: str) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空。")

        chunks = self._retriever.retrieve(normalized_question)
        if not chunks:
            return {
                "question": normalized_question,
                "answer": "未检索到可用知识片段，当前无法基于知识库给出回答。",
                "sources": [],
            }

        answer = self._answer_chain.invoke(
            {
                "question": normalized_question,
                "context": format_context(chunks),
                "evidence_summary": self._summarize_evidence(chunks),
            }
        ).strip()
        answer = self._repair_uncertain_time_answer(
            normalized_question,
            answer,
            chunks,
        )

        return {
            "question": normalized_question,
            "answer": answer,
            "sources": [chunk.to_dict() for chunk in chunks],
        }

    @staticmethod
    def _summarize_evidence(chunks) -> str:
        counts = Counter(chunk.evidence_type for chunk in chunks)
        return (
            "证据概览: "
            f"直接证据 {counts.get('direct', 0)} 条，"
            f"间接证据 {counts.get('indirect', 0)} 条，"
            f"背景证据 {counts.get('background', 0)} 条，"
            f"通用上下文 {counts.get('context', 0)} 条。"
            "优先依据直接证据回答；如果没有直接证据但存在多条间接证据，可以明确标注为“弱推断”；"
            "只有在直接证据和间接证据都不足时，才回答“根据当前知识库无法确认”。"
        )

    @staticmethod
    def _repair_uncertain_time_answer(question, answer: str, chunks) -> str:
        if "根据当前知识库无法确认" not in answer:
            return answer

        if not KnowledgeBaseQAService._is_time_question(question):
            return answer

        fallback = KnowledgeBaseQAService._build_time_fallback_answer(question, chunks)
        return fallback or answer

    @staticmethod
    def _build_time_fallback_answer(question: str, chunks) -> str | None:
        subject = KnowledgeBaseQAService._extract_subject(question)
        subject_candidates = KnowledgeBaseQAService._build_subject_candidates(subject)
        indexed_chunks = [
            (index, chunk)
            for index, chunk in enumerate(chunks, start=1)
            if not subject_candidates or any(candidate in chunk.text for candidate in subject_candidates)
        ]
        if not indexed_chunks:
            return None

        exact_completion = None
        plan_milestone = None
        followup_by_plan = None

        for index, chunk in indexed_chunks:
            text = chunk.text
            date_match = KnowledgeBaseQAService._find_relevant_date(text, subject_candidates)
            has_date = date_match is not None
            has_plan_terms = any(keyword in text for keyword in ("规划", "编制", "设计"))
            has_completion_terms = any(keyword in text for keyword in ("完工", "竣工", "整改", "整治", "完成时限", "完成日期"))

            if "按照规划确定的时间完成" in text:
                followup_by_plan = (index, chunk)

            if has_date and has_completion_terms and not has_plan_terms and exact_completion is None:
                exact_completion = (index, chunk, date_match.group(0))

            if has_date and has_plan_terms and plan_milestone is None:
                plan_milestone = (index, chunk, date_match.group(0))

        subject_text = subject or "该事项"
        if exact_completion is not None:
            ref_index, _, date_text = exact_completion
            return f"{subject_text}的完成时间为{date_text}[{ref_index}]。"

        if plan_milestone is not None and followup_by_plan is not None:
            plan_ref, _, plan_date = plan_milestone
            followup_ref, _ = followup_by_plan
            return (
                f"知识库未给出{subject_text}明确的最终完工日期；"
                f"能直接确认的是相关保护整治详细规划的编制完成时限为{plan_date}[{plan_ref}]，"
                f"后续保护性整治按规划确定的时间完成[{followup_ref}]。"
            )

        if plan_milestone is not None:
            plan_ref, _, plan_date = plan_milestone
            return (
                f"知识库未给出{subject_text}明确的最终完工日期；"
                f"目前能直接确认的时间节点是相关规划编制完成时限为{plan_date}[{plan_ref}]。"
            )

        return None

    @staticmethod
    def _is_time_question(question: str) -> bool:
        return any(
            keyword in question
            for keyword in ("什么时候", "何时", "时间", "日期", "完工", "竣工", "完成")
        )

    @staticmethod
    def _extract_subject(question: str) -> str:
        subject = question
        for keyword in sorted(
            ("什么时候", "何时", "时间", "日期", "完工", "竣工", "完成", "整改完成", "呢", "吗", "请问"),
            key=len,
            reverse=True,
        ):
            subject = subject.replace(keyword, "")

        subject = re.sub(r"[?？,，。.!！:：\s]+", "", subject)
        return subject

    @staticmethod
    def _build_subject_candidates(subject: str) -> list[str]:
        if not subject:
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            normalized = candidate.strip()
            if len(normalized) < 2 or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        add(subject)

        for match in re.finditer(r"[省市区县镇乡村街道]", subject):
            add(subject[match.end():])

        for suffix in ("历史文化街区", "街区", "项目"):
            if subject.endswith(suffix):
                add(subject[: -len(suffix)])

        return candidates

    @staticmethod
    def _find_relevant_date(text: str, subject_candidates: list[str]) -> re.Match[str] | None:
        for subject in subject_candidates:
            closest_match = None
            closest_distance = None
            for subject_match in re.finditer(re.escape(subject), text):
                window = text[subject_match.start(): subject_match.start() + 160]
                date_match = DETAILED_DATE_PATTERN.search(window) or DATE_PATTERN.search(window)
                if date_match is None:
                    continue

                absolute_distance = date_match.start()
                if closest_distance is None or absolute_distance < closest_distance:
                    closest_distance = absolute_distance
                    closest_match = date_match

            if closest_match is not None:
                return closest_match

        return DETAILED_DATE_PATTERN.search(text) or DATE_PATTERN.search(text)


_default_service: KnowledgeBaseQAService | None = None


def ask(question: str) -> dict[str, Any]:
    global _default_service

    if _default_service is None:
        _default_service = KnowledgeBaseQAService()

    return _default_service.ask(question)
