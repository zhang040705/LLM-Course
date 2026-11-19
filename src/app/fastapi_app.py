"""FastAPI 服务端与 SSE 流式接口封装（对应 T15，薄封装复用 RAGService）。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import load_config
from src.rag import RAGResponse, RAGService

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="当前用户提问")
    history: List[ChatMessage] = Field(default_factory=list, description="历史对话消息列表")


def _serialize_rag_response(resp: RAGResponse) -> Dict[str, Any]:
    max_score = max((c.score for c in resp.retrieved_chunks), default=0.0)
    return {
        "answer": resp.answer,
        "citations": [asdict(c) for c in resp.citations],
        "retrieved_count": len(resp.retrieved_chunks),
        "max_score": round(max_score, 4),
        "refused": resp.refused,
        "refusal_reason": resp.refusal_reason,
        "disclaimer": resp.disclaimer,
        "model": resp.model,
        "prompt_tokens": resp.prompt_tokens,
        "completion_tokens": resp.completion_tokens,
        "total_tokens": resp.total_tokens,
        "latency_ms": resp.latency_ms,
        "rewritten_query": resp.rewritten_query,
        "invalid_citations_removed": resp.invalid_citations_removed,
        "context_stats": asdict(resp.context_stats) if resp.context_stats else None,
    }


def create_fastapi_app(
    rag_service: Optional[RAGService] = None,
    config: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """构建 FastAPI 应用（含简洁前端页面、健康检查、REST 与 SSE 流式问答接口）。"""
    cfg = config or load_config()
    proj_cfg = cfg.get("project", {})
    ret_cfg = cfg.get("retrieval", {})
    rag_cfg = cfg.get("rag", {})
    llm_cfg = cfg.get("llm", {})

    app = FastAPI(
        title="中文医疗领域 RAG 问答系统 API",
        version=str(proj_cfg.get("version", "v2-retrieval")),
        description="基于 CMedQA2 的中文医疗领域特定 RAG 问答服务（支持多轮改写、>32k 上下文、引用溯源与拒答）。",
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    service_holder: Dict[str, Optional[RAGService]] = {"instance": rag_service}

    def _get_service() -> RAGService:
        if service_holder["instance"] is None:
            service_holder["instance"] = RAGService.from_config(cfg)
        return service_holder["instance"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": proj_cfg.get("version", "v2-retrieval"),
            "retrieval_mode": ret_cfg.get("mode", "hybrid"),
            "rerank_enabled": bool(ret_cfg.get("rerank_enabled", True)),
            "max_context_tokens": int(llm_cfg.get("max_context_tokens", 65536)),
            "min_similarity_score": float(rag_cfg.get("min_similarity_score", 0.20)),
            "disclaimer": rag_cfg.get("disclaimer", ""),
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> Dict[str, Any]:
        try:
            service = _get_service()
            hist = [m.model_dump() for m in req.history]
            resp = service.answer(req.query, history=hist)
            return _serialize_rag_response(resp)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        hist = [m.model_dump() for m in req.history]

        def event_generator() -> Iterator[str]:
            try:
                service = _get_service()
                for ev in service.stream_answer(req.query, history=hist):
                    if ev["type"] == "done":
                        payload = {
                            "type": "done",
                            "response": _serialize_rag_response(ev["response"]),
                        }
                    else:
                        payload = ev
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception as exc:
                err_payload = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
