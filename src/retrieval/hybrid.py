"""混合检索（BM25 + Dense RRF 融合）与可选 Cross-Encoder 重排模块（对应 T12）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embedder import create_embedder
from src.retrieval.reranker import BaseReranker, create_reranker
from src.retrieval.vectorstore import ChromaVectorStore, RetrievedChunk


class HybridRetriever:
    """Dense 语义检索 + BM25 稀疏检索（RRF 多路融合）+ 可选 Reranker 重排。"""

    def __init__(
        self,
        dense_store: ChromaVectorStore,
        bm25_store: Optional[BM25Retriever] = None,
        reranker: Optional[BaseReranker] = None,
        candidate_k: int = 15,
        rrf_k: int = 60,
        dense_weight: float = 0.65,
        bm25_weight: float = 0.35,
    ) -> None:
        self.dense_store = dense_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def count(self) -> int:
        return self.dense_store.count()

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """执行多路召回、RRF 融合，并在启用重排器时执行 Cross-Encoder 精排。"""
        if not query.strip():
            return []

        fetch_k = max(top_k, self.candidate_k)
        dense_hits = self.dense_store.search(query, top_k=fetch_k)
        bm25_hits = (
            self.bm25_store.search(query, top_k=fetch_k)
            if self.bm25_store is not None and self.bm25_store.count() > 0
            else []
        )

        # RRF 多路融合去重，同时保留原始 dense 余弦分值供无重排时作为相关性分值
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        dense_score_map: Dict[str, float] = {}

        for rank, c in enumerate(dense_hits, start=1):
            fused_scores[c.chunk_id] = fused_scores.get(c.chunk_id, 0.0) + self.dense_weight / (
                self.rrf_k + rank
            )
            chunk_map[c.chunk_id] = c
            dense_score_map[c.chunk_id] = c.score

        for rank, c in enumerate(bm25_hits, start=1):
            fused_scores[c.chunk_id] = fused_scores.get(c.chunk_id, 0.0) + self.bm25_weight / (
                self.rrf_k + rank
            )
            if c.chunk_id not in chunk_map:
                chunk_map[c.chunk_id] = c

        ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        candidates: List[RetrievedChunk] = []
        for cid in ranked_ids[:fetch_k]:
            orig = chunk_map[cid]
            meta = dict(orig.metadata)
            meta["rrf_score"] = round(fused_scores[cid], 6)
            if cid in dense_score_map:
                meta["dense_score"] = dense_score_map[cid]
            candidates.append(
                RetrievedChunk(
                    chunk_id=orig.chunk_id,
                    doc_id=orig.doc_id,
                    text=orig.text,
                    score=dense_score_map.get(cid, round(orig.score * 0.5, 4)),
                    metadata=meta,
                )
            )

        if self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]


def create_retriever(config: Dict[str, Any]) -> Any:
    """根据全局配置构建 Dense 或 Hybrid (+ Reranker) 检索器。"""
    from src.config import PROJECT_ROOT

    ret_cfg = config.get("retrieval", {})
    persist_dir = PROJECT_ROOT / ret_cfg.get("vectorstore_dir", "data/vectorstore")
    collection_name = ret_cfg.get("collection_name", "cmedqa_kb")
    mode = ret_cfg.get("mode", "dense")
    rerank_enabled = bool(ret_cfg.get("rerank_enabled", False))

    embedder = create_embedder(config)
    dense_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=embedder,
    )

    if mode != "hybrid" and not rerank_enabled:
        return dense_store

    bm25_store: Optional[BM25Retriever] = None
    bm25_path = persist_dir / ret_cfg.get("bm25_index_file", "bm25_index.json")
    if mode == "hybrid" and Path(bm25_path).exists():
        bm25_store = BM25Retriever.load(bm25_path)

    reranker: Optional[BaseReranker] = create_reranker(config) if rerank_enabled else None
    return HybridRetriever(
        dense_store=dense_store,
        bm25_store=bm25_store,
        reranker=reranker,
        candidate_k=int(ret_cfg.get("candidate_k", 15)),
        dense_weight=float(ret_cfg.get("dense_weight", 0.65)),
        bm25_weight=float(ret_cfg.get("bm25_weight", 0.35)),
    )
