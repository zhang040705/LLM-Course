"""RAG 核心编排服务（多轮对话、上下文预算、引用校验、拒答与免责声明）。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.retrieval import ChromaVectorStore, RetrievedChunk, create_embedder


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


SYSTEM_PROMPT_TEMPLATE = """你是一名专业、严谨的中文医疗健康科普助手。请严格遵循以下规则回答用户问题：
1. 仅依据下方提供的【参考医疗资料】中的事实进行回答，严禁编造资料中未提及的病因、诊断、药物剂量或检查结论。
2. 在回答的关键结论处，必须使用方括号编号标注引用来源（例如 `[1]`、`[2]`），编号必须与【参考医疗资料】中的编号严格一致。
3. 如果【参考医疗资料】与用户问题无关、信息不足以安全回答，或问题属于非医疗领域，请直接以 `[REFUSE]` 开头拒绝回答，并说明原因（例如：“[REFUSE] 抱歉，当前知识库中未检索到足够的相关医疗依据，建议您咨询专业医生。”）。
4. 回答应条理清晰、通俗易懂，不替代执业医师的面对面诊断。

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
        min_similarity_score: float = 0.45,
        max_context_tokens: int = 32768,
        max_output_tokens: int = 1536,
        disclaimer: str = "【医疗免责声明】本系统回答仅供健康科普参考，不构成临床诊疗依据。如有身体不适，请及时前往正规医院就诊。",
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_history_turns = max_history_turns
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.disclaimer = disclaimer

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
            embedder = create_embedder(config)
            retriever = ChromaVectorStore(
                persist_dir=PROJECT_ROOT / ret_cfg.get("vectorstore_dir", "data/vectorstore"),
                collection_name=ret_cfg.get("collection_name", "cmedqa_kb"),
                embedder=embedder,
            )

        if llm_client is None:
            llm_client = LLMClient.from_config(config)

        return cls(
            retriever=retriever,
            llm_client=llm_client,
            max_history_turns=int(rag_cfg.get("max_history_turns", 6)),
            top_k=int(ret_cfg.get("top_k", 5)),
            min_similarity_score=float(rag_cfg.get("min_similarity_score", 0.45)),
            max_context_tokens=int(llm_cfg.get("max_context_tokens", 32768)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 1536)),
            disclaimer=str(
                rag_cfg.get(
                    "disclaimer",
                    "【医疗免责声明】本系统回答仅供健康科普参考，不构成临床诊疗依据。如有身体不适，请及时前往正规医院就诊。",
                )
            ),
        )

    def _format_context(self, chunks: List[RetrievedChunk], token_budget: int) -> str:
        """按 Token 预算拼装带编号的检索证据。"""
        blocks: List[str] = []
        used_tokens = 0
        for idx, chunk in enumerate(chunks, start=1):
            block = f"[{idx}] (相似度: {chunk.score:.3f}, 来源: {chunk.doc_id})\n{chunk.text}"
            cost = LLMClient.estimate_tokens(block)
            if used_tokens + cost > token_budget and blocks:
                break
            blocks.append(block)
            used_tokens += cost
        return "\n\n".join(blocks)

    def _trim_history(
        self,
        history: Optional[List[Dict[str, str]]],
        token_budget: int,
    ) -> List[Dict[str, str]]:
        """保留最近 N 轮对话并控制在 Token 预算内。"""
        if not history or self.max_history_turns <= 0:
            return []

        recent = history[-(self.max_history_turns * 2) :]
        kept: List[Dict[str, str]] = []
        used = 0
        for msg in reversed(recent):
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role not in ("user", "assistant") or not content:
                continue
            cost = LLMClient.estimate_tokens(content)
            if used + cost > token_budget and kept:
                break
            kept.append({"role": role, "content": content})
            used += cost
        kept.reverse()
        return kept

    @staticmethod
    def _extract_and_clean_citations(
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> tuple[str, List[Citation]]:
        """校验回答中的 [n] 引用编号，过滤虚构编号并返回对应 Citation 列表。"""
        valid_max = len(chunks)
        cited_order: List[int] = []

        def _replace(match: re.Match[str]) -> str:
            num = int(match.group(1))
            if 1 <= num <= valid_max:
                if num not in cited_order:
                    cited_order.append(num)
                return f"[{num}]"
            return ""

        cleaned_answer = re.sub(r"\[(\d+)\]", _replace, answer).strip()
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
        return cleaned_answer, citations

    def _with_disclaimer(self, text: str) -> str:
        if self.disclaimer and self.disclaimer not in text:
            return f"{text.strip()}\n\n{self.disclaimer}"
        return text.strip()

    def answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGResponse:
        """执行端到端 RAG 问答：检索 → 拒答判定 → 生成 → 引用校验 → 免责注入。"""
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

        chunks = self.retriever.search(clean_query, top_k=self.top_k)
        max_score = max((c.score for c in chunks), default=0.0)

        # 1. 前置相似度阈值拒答（无需消耗 LLM Token）
        if not chunks or max_score < self.min_similarity_score:
            return RAGResponse(
                answer=self._with_disclaimer(DEFAULT_REFUSAL_TEXT),
                citations=[],
                retrieved_chunks=chunks,
                refused=True,
                refusal_reason=f"low_similarity({max_score:.4f}<{self.min_similarity_score})",
                disclaimer=self.disclaimer,
                latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
            )

        if self.llm_client is None:
            raise RuntimeError("LLMClient 未初始化，无法执行模型生成。")

        # 2. 按 >32k 上下文预算拼装历史与检索证据
        available_input_tokens = max(1024, self.max_context_tokens - self.max_output_tokens - 512)
        context_budget = int(available_input_tokens * 0.7)
        history_budget = available_input_tokens - context_budget

        context_str = self._format_context(chunks, token_budget=context_budget)
        trimmed_history = self._trim_history(history, token_budget=history_budget)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context_str)}
        ]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": clean_query})

        llm_resp = self.llm_client.chat(messages)
        raw_text = llm_resp.content

        # 3. 模型级拒答检测与引用后置校验
        refused = False
        refusal_reason = ""
        if raw_text.startswith("[REFUSE]"):
            refused = True
            refusal_reason = "llm_refusal"
            raw_text = raw_text.removeprefix("[REFUSE]").strip()

        cleaned_answer, citations = self._extract_and_clean_citations(raw_text, chunks)
        if refused:
            citations = []

        total_latency = round((time.perf_counter() - start) * 1000.0, 2)
        return RAGResponse(
            answer=self._with_disclaimer(cleaned_answer),
            citations=citations,
            retrieved_chunks=chunks,
            refused=refused,
            refusal_reason=refusal_reason,
            disclaimer=self.disclaimer,
            model=llm_resp.model,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            total_tokens=llm_resp.total_tokens,
            latency_ms=total_latency,
        )
