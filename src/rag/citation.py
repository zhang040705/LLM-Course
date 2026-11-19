"""引用标记规范化与真实性后置校验模块（对应 T10）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.retrieval.vectorstore import RetrievedChunk


@dataclass
class Citation:
    """单条可解释引用来源结构。"""

    index: int
    chunk_id: str
    doc_id: str
    question: str
    text: str
    score: float
    source: str = "cMedQA2"


@dataclass
class CitationCheckResult:
    """引用后置校验与提取结果。"""

    cleaned_answer: str
    citations: List[Citation] = field(default_factory=list)
    invalid_indices: List[int] = field(default_factory=list)
    ungrounded_indices: List[int] = field(default_factory=list)


def normalize_citation_markers(text: str) -> str:
    """将全角【1】或组合引用 [1, 2]、[1、2] 统一规范化为标准 [1][2] 格式。"""
    if not text:
        return ""

    # 1. 将全角【1】、【1、2】替换为半角方括号格式
    def _replace_fullwidth(match: re.Match[str]) -> str:
        inner = match.group(1)
        nums = re.findall(r"\d+", inner)
        return "".join(f"[{n}]" for n in nums)

    normalized = re.sub(
        r"【(\s*\d+(?:\s*[,，、\s]\s*\d+)*\s*)】",
        _replace_fullwidth,
        text,
    )

    # 2. 将组合引用 [1, 2] / [1、2] / [1，2] 拆分为 [1][2]
    def _replace_multi(match: re.Match[str]) -> str:
        nums = re.findall(r"\d+", match.group(1))
        return "".join(f"[{n}]" for n in nums)

    normalized = re.sub(
        r"\[(\s*\d+(?:\s*[,，、\s]\s*\d+)+\s*)\]",
        _replace_multi,
        normalized,
    )
    return normalized


def _strip_markers_and_ws(text: str) -> str:
    cleaned = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", "", cleaned)


def verify_and_extract_citations(
    answer: str,
    chunks: List[RetrievedChunk],
    min_overlap_chars: int = 4,
) -> CitationCheckResult:
    """校验回答中的引用编号真实性，过滤虚构编号并检测无支撑引用。"""
    normalized_text = normalize_citation_markers(answer)
    valid_max = len(chunks)
    cited_order: List[int] = []
    invalid_indices: List[int] = []

    def _replace(match: re.Match[str]) -> str:
        num = int(match.group(1))
        if 1 <= num <= valid_max:
            if num not in cited_order:
                cited_order.append(num)
            return f"[{num}]"
        if num not in invalid_indices:
            invalid_indices.append(num)
        return ""

    cleaned_answer = re.sub(r"\[(\d+)\]", _replace, normalized_text).strip()
    citations = [
        Citation(
            index=num,
            chunk_id=chunks[num - 1].chunk_id,
            doc_id=chunks[num - 1].doc_id,
            question=str(chunks[num - 1].metadata.get("question", "")),
            text=chunks[num - 1].text,
            score=chunks[num - 1].score,
            source=str(chunks[num - 1].metadata.get("source", "cMedQA2")),
        )
        for num in cited_order
    ]

    # 检查有效引用是否与回答正文存在基本字符重叠（识别疑似“空标/错标”引用）
    answer_chars = set(_strip_markers_and_ws(cleaned_answer))
    ungrounded_indices: List[int] = []
    for c in citations:
        chunk_chars = set(_strip_markers_and_ws(c.text))
        if len(answer_chars & chunk_chars) < min_overlap_chars:
            ungrounded_indices.append(c.index)

    return CitationCheckResult(
        cleaned_answer=cleaned_answer,
        citations=citations,
        invalid_indices=invalid_indices,
        ungrounded_indices=ungrounded_indices,
    )
