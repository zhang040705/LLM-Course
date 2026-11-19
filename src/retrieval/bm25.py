"""零依赖中文 Okapi BM25 稀疏检索模块（支持本地 JSON 持久化，对应 T12）。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.vectorstore import RetrievedChunk

_EN_NUM_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize_zh(text: str) -> List[str]:
    """面向中文医疗文本的轻量分词器（汉字 unigram + bigram + 英文/数字词元）。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = _EN_NUM_RE.findall(lowered)
    for run in _CJK_RUN_RE.findall(lowered):
        n = len(run)
        # 单字保留召回，二字 bigram 强化专业医疗术语精确匹配（如“阿莫西林”“四维彩超”）
        tokens.extend(run)
        if n >= 2:
            tokens.extend(run[i : i + 2] for i in range(n - 1))
    return tokens


class BM25Retriever:
    """轻量持久化 Okapi BM25 检索器。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[Dict[str, Any]] = []
        self.doc_freqs: List[Counter[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted: Dict[str, List[int]] = {}

    def count(self) -> int:
        return len(self.docs)

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """从分块字典列表构建 BM25 倒排索引。"""
        self.docs = []
        self.doc_freqs = []
        self.doc_lens = []
        self.idf = {}
        self.inverted = {}

        df: Counter[str] = Counter()
        for idx, item in enumerate(chunks):
            doc_rec = {
                "chunk_id": str(item["chunk_id"]),
                "doc_id": str(item.get("doc_id", "")),
                "text": str(item["text"]),
                "metadata": {
                    "doc_id": str(item.get("doc_id", "")),
                    "question_id": str(item.get("question_id", "")),
                    "question": str(item.get("question", ""))[:200],
                    "source": str(item.get("source", "cMedQA2")),
                    "chunk_index": int(item.get("chunk_index", 0)),
                },
            }
            self.docs.append(doc_rec)
            toks = tokenize_zh(doc_rec["text"])
            tf = Counter(toks)
            self.doc_freqs.append(tf)
            self.doc_lens.append(len(toks))
            for tok in tf:
                df[tok] += 1
                self.inverted.setdefault(tok, []).append(idx)

        n_docs = len(self.docs)
        self.avgdl = (sum(self.doc_lens) / n_docs) if n_docs > 0 else 0.0
        for tok, freq in df.items():
            self.idf[tok] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        return n_docs

    def save(self, path: str | Path) -> None:
        """持久化分块原文与元数据为 JSON（加载时快速重建内存倒排表）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "chunks": [
                {
                    "chunk_id": d["chunk_id"],
                    "doc_id": d["doc_id"],
                    "text": d["text"],
                    **d["metadata"],
                }
                for d in self.docs
            ],
        }
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        """从持久化 JSON 加载并恢复 BM25 索引。"""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        inst = cls(k1=float(payload.get("k1", 1.5)), b=float(payload.get("b", 0.75)))
        inst.build_from_chunks(payload.get("chunks", []))
        return inst

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """执行 BM25 稀疏检索，返回按归一化分值排序的 Top-K 分块。"""
        if not self.docs or not query.strip():
            return []

        q_tokens = tokenize_zh(query)
        if not q_tokens:
            return []

        scores: Dict[int, float] = {}
        avgdl = self.avgdl or 1.0
        for tok in q_tokens:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for doc_idx in self.inverted.get(tok, ()):
                tf = self.doc_freqs[doc_idx][tok]
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1.0) / denom)

        if not scores:
            return []

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_raw = top_items[0][1] if top_items[0][1] > 0 else 1.0

        results: List[RetrievedChunk] = []
        for doc_idx, raw_score in top_items:
            d = self.docs[doc_idx]
            norm_score = min(1.0, raw_score / (max_raw + 1e-6))
            meta = dict(d["metadata"])
            meta["bm25_raw"] = round(raw_score, 4)
            results.append(
                RetrievedChunk(
                    chunk_id=d["chunk_id"],
                    doc_id=d["doc_id"],
                    text=d["text"],
                    score=round(norm_score, 4),
                    metadata=meta,
                )
            )
        return results
