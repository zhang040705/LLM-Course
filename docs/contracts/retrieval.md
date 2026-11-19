# 契约：检索与向量化层（`src/retrieval/`）

- **关联任务**：`T03`（本地向量库构建）、`T11`（Embedding 升级与云端对比）、`T12`（混合检索与重排）
- **实现文件**：`src/retrieval/embedder.py`、`src/retrieval/vectorstore.py`、`src/retrieval/bm25.py`、`src/retrieval/reranker.py`、`src/retrieval/hybrid.py`

## 1. Embedder 契约
- `BaseEmbedder.embed_documents(texts: list[str]) -> list[list[float]]`
- `BaseEmbedder.embed_query(text: str) -> list[float]`
- 工厂方法 `create_embedder(config)`：
  - `provider == "local"`：使用 `LocalONNXEmbedder`（支持 `Xenova/bge-small-zh-v1.5` 与升级版 `Xenova/bge-base-zh-v1.5` 量化 ONNX，L2 归一化）。
  - `provider == "dashscope"`：使用 `DashScopeEmbedder`（需 `.env` 提供 `DASHSCOPE_API_KEY`）。

## 2. 检索与重排契约
- `RetrievedChunk(chunk_id, doc_id, text, score, metadata)`：
  - Dense 模式下 `score` 为归一化余弦相似度 `[0.0, 1.0]`。
  - Hybrid + Reranker 模式下，`score` 为 Cross-Encoder（`Xenova/bge-reranker-base`）经 Sigmoid 映射后的相关性概率 `(0.0, 1.0)`，原始 `dense_score`、`rrf_score` 保留在 `metadata` 中。
  - `metadata` 必须保留 `doc_id`、`question_id`、`question`、`source`、`chunk_index`，支撑引用溯源。
- `create_retriever(config)`：
  - `retrieval.mode == "dense"` 且 `rerank_enabled == false`：返回 `ChromaVectorStore`。
  - `retrieval.mode == "hybrid"` 或 `rerank_enabled == true`：返回 `HybridRetriever`（Dense + BM25 RRF 多路融合 + 可选 `LocalONNXReranker` 精排）。
