"""Cross-Encoder 重排（Reranker）抽象与本地 ONNX 实现（对应 T12）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from src.retrieval.vectorstore import RetrievedChunk


class BaseReranker(ABC):
    """重排器统一抽象基类。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """对候选检索分块进行交叉打分并按相关性降序返回 Top-K。"""


class LocalONNXReranker(BaseReranker):
    """基于 ONNX Runtime 的本地中文 Cross-Encoder 重排器（默认 Xenova/bge-reranker-base）。"""

    def __init__(
        self,
        model_id: str = "Xenova/bge-reranker-base",
        onnx_file: str = "onnx/model_quantized.onnx",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size

        model_path = hf_hub_download(repo_id=model_id, filename=onnx_file)
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(length=None)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def _score_pairs(self, query: str, texts: List[str]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch_pairs = [(query, t if t.strip() else " ") for t in texts[i : i + self.batch_size]]
            encodings = self._tokenizer.encode_batch(batch_pairs)
            input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

            feeds: Dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

            outputs = self._session.run(None, feeds)
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            scores.extend(float(p) for p in probs)
        return scores

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not chunks or not query.strip():
            return []

        probs = self._score_pairs(query, [c.text for c in chunks])
        reranked: List[RetrievedChunk] = []
        for chunk, prob in zip(chunks, probs):
            meta = dict(chunk.metadata)
            meta["pre_rerank_score"] = chunk.score
            meta["rerank_score"] = round(prob, 4)
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=round(prob, 4),
                    metadata=meta,
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


def create_reranker(config: Dict[str, Any]) -> BaseReranker:
    """根据全局配置构建 Reranker 实例。"""
    ret_cfg = config.get("retrieval", {})
    return LocalONNXReranker(
        model_id=ret_cfg.get("rerank_model", "Xenova/bge-reranker-base"),
        onnx_file=ret_cfg.get("rerank_onnx_file", "onnx/model_quantized.onnx"),
        batch_size=int(ret_cfg.get("rerank_batch_size", 16)),
    )
