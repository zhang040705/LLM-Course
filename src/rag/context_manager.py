"""长上下文（>32k tokens）预算管理与对话历史规范化模块（对应 T09）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.retrieval.vectorstore import RetrievedChunk


@dataclass
class ContextStats:
    """单次请求的上下文 Token 预算与截断统计信息。"""

    estimated_prompt_tokens: int
    context_tokens: int
    history_tokens: int
    included_chunks: int
    dropped_chunks: int
    included_history_msgs: int
    dropped_history_msgs: int
    truncated: bool
    max_context_tokens: int


def _extract_message_text(content: Any) -> str:
    """兼容字符串与 Gradio 多段消息块列表格式，提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


def normalize_history(
    history: Optional[List[Dict[str, Any]]],
    disclaimer: str = "",
) -> List[Dict[str, str]]:
    """规范化对话历史并去除助手回复中重复拼接的医疗免责声明。"""
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if role not in ("user", "assistant"):
            continue
        text = _extract_message_text(msg.get("content")).strip()
        if role == "assistant" and disclaimer:
            text = text.replace(disclaimer, "").strip()
        if text:
            normalized.append({"role": role, "content": text})
    return normalized


class ContextBudgetManager:
    """证据优先、动态共享剩余 Token 预算的上下文管理器（支持 >32k tokens）。"""

    def __init__(
        self,
        max_context_tokens: int = 65536,
        max_output_tokens: int = 1536,
        reserve_overhead_tokens: int = 512,
        min_context_ratio: float = 0.5,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.reserve_overhead_tokens = reserve_overhead_tokens
        self.min_context_ratio = min_context_ratio

    @staticmethod
    def _truncate_text_to_budget(text: str, token_budget: int) -> str:
        """按字符比例安全截断超长文本，确保不超过 token_budget。"""
        if token_budget <= 0 or not text:
            return ""
        est = LLMClient.estimate_tokens(text)
        if est <= token_budget:
            return text
        ratio = max(0.05, (token_budget - 8) / max(1, est))
        cut = max(1, int(len(text) * ratio))
        truncated = text[:cut].rstrip()
        while truncated and LLMClient.estimate_tokens(truncated + "...") > token_budget:
            cut = int(cut * 0.85)
            if cut <= 0:
                break
            truncated = text[:cut].rstrip()
        return (truncated + "...") if truncated else text[: min(len(text), 64)]

    def build(
        self,
        chunks: List[RetrievedChunk],
        history: Optional[List[Dict[str, Any]]],
        query: str,
        system_prompt_overhead: int = 256,
        max_history_turns: int = 6,
        disclaimer: str = "",
    ) -> tuple[str, List[Dict[str, str]], ContextStats]:
        """按预算拼装检索证据与历史对话，保护 Top-1 证据不被静默丢弃。"""
        query_tokens = LLMClient.estimate_tokens(query)
        available_input_tokens = max(
            1024,
            self.max_context_tokens
            - self.max_output_tokens
            - self.reserve_overhead_tokens
            - system_prompt_overhead
            - query_tokens,
        )

        # 1. 预先规范化并按轮数截取候选历史，以计算证据与历史的动态预算分配
        norm_history = normalize_history(history, disclaimer=disclaimer)
        if max_history_turns > 0:
            candidate_history = norm_history[-(max_history_turns * 2) :]
            dropped_by_turns = max(0, len(norm_history) - len(candidate_history))
        else:
            candidate_history = norm_history
            dropped_by_turns = 0

        history_needed = sum(LLMClient.estimate_tokens(m["content"]) for m in candidate_history)
        min_evidence_budget = int(available_input_tokens * self.min_context_ratio)
        reserved_for_history = min(history_needed, max(0, available_input_tokens - min_evidence_budget))
        evidence_budget = max(min_evidence_budget, available_input_tokens - reserved_for_history)

        # 2. 证据优先装填：Top-1 永不静默丢弃
        blocks: List[str] = []
        context_tokens = 0
        included_chunks = 0
        dropped_chunks = 0
        truncated = False

        for idx, chunk in enumerate(chunks, start=1):
            header = f"[{idx}] (相似度: {chunk.score:.3f}, 来源: {chunk.doc_id})\n"
            block = f"{header}{chunk.text}"
            cost = LLMClient.estimate_tokens(block)

            if context_tokens + cost <= evidence_budget:
                blocks.append(block)
                context_tokens += cost
                included_chunks += 1
            elif idx == 1:
                header_cost = LLMClient.estimate_tokens(header)
                text_budget = max(64, evidence_budget - header_cost)
                clipped = self._truncate_text_to_budget(chunk.text, text_budget)
                clipped_block = f"{header}{clipped}"
                blocks.append(clipped_block)
                context_tokens += LLMClient.estimate_tokens(clipped_block)
                included_chunks += 1
                truncated = True
            else:
                dropped_chunks = len(chunks) - idx + 1
                truncated = True
                break

        context_str = "\n\n".join(blocks)

        # 3. 剩余全部可用预算动态分配给历史消息（从最新消息倒序装填）
        history_budget = max(0, available_input_tokens - context_tokens)
        kept_history: List[Dict[str, str]] = []
        history_tokens = 0
        dropped_by_budget = 0

        for msg in reversed(candidate_history):
            cost = LLMClient.estimate_tokens(msg["content"])
            if history_tokens + cost <= history_budget:
                kept_history.append(msg)
                history_tokens += cost
            else:
                dropped_by_budget = len(candidate_history) - len(kept_history)
                truncated = True
                break

        kept_history.reverse()
        dropped_history_msgs = dropped_by_turns + dropped_by_budget
        estimated_prompt_tokens = system_prompt_overhead + context_tokens + history_tokens + query_tokens

        stats = ContextStats(
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_tokens=context_tokens,
            history_tokens=history_tokens,
            included_chunks=included_chunks,
            dropped_chunks=dropped_chunks,
            included_history_msgs=len(kept_history),
            dropped_history_msgs=dropped_history_msgs,
            truncated=truncated,
            max_context_tokens=self.max_context_tokens,
        )
        return context_str, kept_history, stats
