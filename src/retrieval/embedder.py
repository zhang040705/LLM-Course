"""文本向量化（Embedding）统一抽象与本地/云端实现。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseEmbedder(ABC):
    """Embedding 模型抽象基类，支持本地与云端实现无缝切换。"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档文本列表。"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """编码单条查询文本。"""


class LocalONNXEmbedder(BaseEmbedder):
    """基于 ONNX Runtime + Tokenizers 的本地轻量中文 Embedding 实现。

    默认加载 Xenova/bge-small-zh-v1.5（约 24MB 量化版或 95MB 全量版），
    无需安装 PyTorch / sentence-transformers 即可在 CPU 上快速离线运行。
    """

    DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_id: str = "Xenova/bge-small-zh-v1.5",
        onnx_file: str = "onnx/model_quantized.onnx",
        batch_size: int = 64,
        max_length: int = 512,
        query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction

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

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        feeds: Dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, feeds)
        # BGE 系列使用 [CLS] token (index 0) 池化并进行 L2 归一化
        cls_embeddings = outputs[0][:, 0, :]
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = cls_embeddings / norms
        return normalized.astype(np.float32).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            results.extend(self._encode_batch(batch))
        return results

    def embed_query(self, text: str) -> List[float]:
        query_text = f"{self.query_instruction}{text}" if self.query_instruction else text
        return self.embed_documents([query_text])[0]


class DashScopeEmbedder(BaseEmbedder):
    """阿里云百炼 DashScope OpenAI 兼容 Embedding 适配器（V2 对比实验用）。"""

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        batch_size: int = 10,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not key or key == "your_dashscope_api_key_here":
            raise ValueError("未检测到有效的 DASHSCOPE_API_KEY，请在 .env 中配置。")
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            results.extend([item.embedding for item in sorted_data])
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def create_embedder(config: Dict[str, Any]) -> BaseEmbedder:
    """根据全局配置创建对应的 Embedder 实例。"""
    emb_cfg = config.get("embedding", {})
    provider = emb_cfg.get("provider", "local")
    batch_size = int(emb_cfg.get("batch_size", 64))

    if provider == "local":
        model_id = emb_cfg.get("onnx_model", "Xenova/bge-small-zh-v1.5")
        onnx_file = emb_cfg.get("onnx_file", "onnx/model_quantized.onnx")
        return LocalONNXEmbedder(
            model_id=model_id,
            onnx_file=onnx_file,
            batch_size=batch_size,
        )

    if provider == "dashscope":
        return DashScopeEmbedder(
            model=emb_cfg.get("dashscope_model", "text-embedding-v3"),
            batch_size=min(batch_size, 10),
        )

    raise ValueError(f"不支持的 embedding provider: {provider}（可选: local, dashscope）")
