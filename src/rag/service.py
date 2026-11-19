"""RAG 核心编排服务（多轮改写、长上下文预算、引用校验、拒答与免责声明）。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from src.llm.client import LLMClient
from src.rag.citation import Citation, verify_and_extract_citations
from src.rag.context_manager import ContextBudgetManager, ContextStats
from src.rag.rewriter import QueryRewriter
from src.retrieval import RetrievedChunk, create_retriever


@dataclass
class RAGResponse:
    """RAG 服务统一返回结构（供 Gradio / FastAPI / 评测脚本共用）。"""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    disclaimer: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    rewritten_query: str = ""
    invalid_citations_removed: List[int] = field(default_factory=list)
    context_stats: Optional[ContextStats] = None


SYSTEM_PROMPT_TEMPLATE = """你是一名专业、严谨的中文医疗健康科普助手。请严格遵循以下规则回答用户问题：
1. 仅依据下方提供的【参考医疗资料】中的事实进行回答，严禁编造资料中未提及的病因、诊断、药物剂量、数值或检查结论。
2. 回答必须直接、精炼地回应用户当前提问，并在每条关键结论后紧跟半角方括号引用编号（例如 `[1]`、`[2]`），编号必须与【参考医疗资料】中的序号严格一致。不要使用【1】或 [1, 2] 格式。
3. 若【参考医疗资料】无法充分回答用户问题、与用户当前意图不符，或问题属于非医疗领域，请直接以 `[REFUSE]` 开头拒绝回答，并说明原因（例如：“[REFUSE] 抱歉，当前知识库中未检索到足够的相关医疗依据，建议您咨询专业医生。”）。
4. 保持专业客观，不替代执业医师的面对面诊断。

【参考医疗资料】
{context}"""

DEFAULT_REFUSAL_TEXT = (
    "抱歉，当前医疗知识库中未检索到与您问题高度相关的可靠依据。"
    "为避免误导，我无法给出确切回答。建议您补充更多症状描述，或前往正规医院咨询专业医生。"
)


class RAGService:
    """医疗领域 RAG 核心编排服务。"""

    def __init__(
        self,
        retriever: Any,
        llm_client: Optional[LLMClient],
        max_history_turns: int = 6,
        top_k: int = 5,
        min_similarity_score: float = 0.52,
        max_context_tokens: int = 65536,
        max_output_tokens: int = 1536,
        disclaimer: str = "【医疗免责声明】本系统回答仅供健康科普参考，不构成临床诊疗依据。如有身体不适，请及时前往正规医院就诊。",
        query_rewrite_enabled: bool = False,
        min_chunk_score: float = 0.50,
        require_citation: bool = True,
        context_manager: Optional[ContextBudgetManager] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_history_turns = max_history_turns
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.disclaimer = disclaimer
        self.query_rewrite_enabled = query_rewrite_enabled
        self.min_chunk_score = min_chunk_score
        self.require_citation = require_citation

        self.context_manager = context_manager or ContextBudgetManager(
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        )
        self.query_rewriter = query_rewriter or QueryRewriter(
            llm_client=llm_client,
            disclaimer=disclaimer,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        retriever: Any = None,
        llm_client: Optional[LLMClient] = None,
    ) -> "RAGService":
        """从全局配置构建 RAGService 实例。"""
        from src.config import PROJECT_ROOT

        ret_cfg = config.get("retrieval", {})
        rag_cfg = config.get("rag", {})
        llm_cfg = config.get("llm", {})

        if retriever is None:
            retriever = create_retriever(config)

        if llm_client is None:
            llm_client = LLMClient.from_config(config)

        return cls(
            retriever=retriever,
            llm_client=llm_client,
            max_history_turns=int(rag_cfg.get("max_history_turns", 6)),
            top_k=int(ret_cfg.get("top_k", 5)),
            min_similarity_score=float(rag_cfg.get("min_similarity_score", 0.52)),
            max_context_tokens=int(llm_cfg.get("max_context_tokens", 65536)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 1536)),
            disclaimer=str(
                rag_cfg.get(
                    "disclaimer",
                    "【医疗免责声明】本系统回答仅供健康科普参考，不构成临床诊疗依据。如有身体不适，请及时前往正规医院就诊。",
                )
            ),
            query_rewrite_enabled=bool(rag_cfg.get("query_rewrite_enabled", False)),
            min_chunk_score=float(rag_cfg.get("min_chunk_score", 0.50)),
            require_citation=bool(rag_cfg.get("require_citation", True)),
        )

    @staticmethod
    def _extract_and_clean_citations(
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> tuple[str, List[Citation]]:
        """兼容方法：校验回答中的 [n] 引用编号并返回清洗后文本与 Citation 列表。"""
        result = verify_and_extract_citations(answer, chunks)
        return result.cleaned_answer, result.citations

    def _with_disclaimer(self, text: str) -> str:
        if self.disclaimer and self.disclaimer not in text:
            return f"{text.strip()}\n\n{self.disclaimer}"
        return text.strip()

    def answer(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGResponse:
        """执行端到端 RAG 问答：改写 → 检索 → 阈值拒答 → 上下文预算 → 生成 → 引用校验。"""
        start = time.perf_counter()
        clean_query = (query or "").strip()
        if not clean_query:
            return RAGResponse(
                answer=self._with_disclaimer("请输入具体的医疗健康问题。"),
                refused=True,
                refusal_reason="empty_query",
                disclaimer=self.disclaimer,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            )

        # 1. 多轮 Query 改写（仅在开启开关且存在对话历史时触发）
        rewritten_query = ""
        search_query = clean_query
        if self.query_rewrite_enabled and history:
            rewritten_query = self.query_rewriter.rewrite(clean_query, history=history)
            if rewritten_query:
                search_query = rewritten_query

        # 2. 检索 Top-K 证据
        raw_chunks = self.retriever.search(search_query, top_k=self.top_k)
        max_score = max((c.score for c in raw_chunks), default=0.0)

        # 3. 前置相似度阈值拒答（零 LLM Token 消耗）
        if not raw_chunks or max_score < self.min_similarity_score:
            return RAGResponse(
                answer=self._with_disclaimer(DEFAULT_REFUSAL_TEXT),
                citations=[],
                retrieved_chunks=raw_chunks,
                refused=True,
                refusal_reason=f"low_similarity({max_score:.4f}<{self.min_similarity_score})",
                disclaimer=self.disclaimer,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                rewritten_query=rewritten_query,
            )

        if self.llm_client is None:
            raise RuntimeError("LLMClient 未初始化，无法执行模型生成。")

        # 4. 过滤低分噪声片段（始终至少保留 Top-1 证据）
        filtered_chunks = [c for c in raw_chunks if c.score >= self.min_chunk_score]
        if not filtered_chunks:
            filtered_chunks = [raw_chunks[0]]

        # 5. >32k 上下文预算拼装与保护
        context_str, trimmed_history, context_stats = self.context_manager.build(
            chunks=filtered_chunks,
            history=history,
            query=clean_query,
            max_history_turns=self.max_history_turns,
            disclaimer=self.disclaimer,
        )
        active_chunks = filtered_chunks[: max(1, context_stats.included_chunks)]

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context_str)}
        ]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": clean_query})

        llm_resp = self.llm_client.chat(messages)
        raw_text = llm_resp.content

        # 6. 模型级拒答检测与引用真实性后置校验
        refused = False
        refusal_reason = ""
        if raw_text.startswith("[REFUSE]"):
            refused = True
            refusal_reason = "llm_refusal"
            raw_text = raw_text.removeprefix("[REFUSE]").strip()

        cite_check = verify_and_extract_citations(raw_text, active_chunks)
        cleaned_answer = cite_check.cleaned_answer
        citations = [] if refused else cite_check.citations

        # 强制引用守卫：非拒答回答若完全缺少有效引用编号，视为无依据并触发后置拒答
        if not refused and self.require_citation and not citations:
            refused = True
            refusal_reason = "no_valid_citation"
            cleaned_answer = DEFAULT_REFUSAL_TEXT
            citations = []

        total_latency = round((time.perf_counter() - start) * 1000.0, 2)
        return RAGResponse(
            answer=self._with_disclaimer(cleaned_answer),
            citations=citations,
            retrieved_chunks=active_chunks,
            refused=refused,
            refusal_reason=refusal_reason,
            disclaimer=self.disclaimer,
            model=llm_resp.model,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            latency_ms=total_latency,
            rewritten_query=rewritten_query,
            invalid_citations_removed=cite_check.invalid_indices,
            context_stats=context_stats,
        )

    def stream_answer(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """流式 RAG 问答生成器（供 FastAPI SSE 使用，复用核心检索/改写/拒答/引用逻辑）。

        事件协议：
        - {"type": "meta", "rewritten_query": str, "retrieved_count": int}
        - {"type": "delta", "content": str}
        - {"type": "done", "response": RAGResponse}
        """
        start = time.perf_counter()
        clean_query = (query or "").strip()
        if not clean_query:
            resp = RAGResponse(
                answer=self._with_disclaimer("请输入具体的医疗健康问题。"),
                refused=True,
                refusal_reason="empty_query",
                disclaimer=self.disclaimer,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            )
            yield {"type": "done", "response": resp}
            return

        rewritten_query = ""
        search_query = clean_query
        if self.query_rewrite_enabled and history:
            rewritten_query = self.query_rewriter.rewrite(clean_query, history=history)
            if rewritten_query:
                search_query = rewritten_query

        raw_chunks = self.retriever.search(search_query, top_k=self.top_k)
        max_score = max((c.score for c in raw_chunks), default=0.0)
        yield {
            "type": "meta",
            "rewritten_query": rewritten_query,
            "retrieved_count": len(raw_chunks),
        }

        if not raw_chunks or max_score < self.min_similarity_score:
            resp = RAGResponse(
                answer=self._with_disclaimer(DEFAULT_REFUSAL_TEXT),
                citations=[],
                retrieved_chunks=raw_chunks,
                refused=True,
                refusal_reason=f"low_similarity({max_score:.4f}<{self.min_similarity_score})",
                disclaimer=self.disclaimer,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
                rewritten_query=rewritten_query,
            )
            yield {"type": "delta", "content": resp.answer}
            yield {"type": "done", "response": resp}
            return

        if self.llm_client is None:
            raise RuntimeError("LLMClient 未初始化，无法执行模型生成。")

        filtered_chunks = [c for c in raw_chunks if c.score >= self.min_chunk_score]
        if not filtered_chunks:
            filtered_chunks = [raw_chunks[0]]

        context_str, trimmed_history, context_stats = self.context_manager.build(
            chunks=filtered_chunks,
            history=history,
            query=clean_query,
            max_history_turns=self.max_history_turns,
            disclaimer=self.disclaimer,
        )
        active_chunks = filtered_chunks[: max(1, context_stats.included_chunks)]

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context_str)}
        ]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": clean_query})

        collected: List[str] = []
        prefix_buf = ""
        prefix_checked = False
        for piece in self.llm_client.stream_chat(messages):
            collected.append(piece)
            if not prefix_checked:
                prefix_buf += piece
                if len(prefix_buf) >= 10:
                    prefix_checked = True
                    emit = prefix_buf.removeprefix("[REFUSE]").lstrip()
                    if emit:
                        yield {"type": "delta", "content": emit}
            else:
                yield {"type": "delta", "content": piece}

        if not prefix_checked and prefix_buf:
            emit = prefix_buf.removeprefix("[REFUSE]").lstrip()
            if emit:
                yield {"type": "delta", "content": emit}

        raw_text = "".join(collected).strip()
        refused = False
        refusal_reason = ""
        if raw_text.startswith("[REFUSE]"):
            refused = True
            refusal_reason = "llm_refusal"
            raw_text = raw_text.removeprefix("[REFUSE]").strip()

        cite_check = verify_and_extract_citations(raw_text, active_chunks)
        cleaned_answer = cite_check.cleaned_answer
        citations = [] if refused else cite_check.citations

        if not refused and self.require_citation and not citations:
            refused = True
            refusal_reason = "no_valid_citation"
            cleaned_answer = DEFAULT_REFUSAL_TEXT
            citations = []

        resp = RAGResponse(
            answer=self._with_disclaimer(cleaned_answer),
            citations=citations,
            retrieved_chunks=active_chunks,
            refused=refused,
            refusal_reason=refusal_reason,
            disclaimer=self.disclaimer,
            model=self.llm_client.model,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            rewritten_query=rewritten_query,
            invalid_citations_removed=cite_check.invalid_indices,
            context_stats=context_stats,
        )
        yield {"type": "done", "response": resp}
