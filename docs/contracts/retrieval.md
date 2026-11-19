# 契约：检索与向量化层（`src/retrieval/`）

- **关联任务**：`T03`（本地向量库构建）、`T11`（Embedding 对比）、`T12`（混合检索与重排）
- **实现文件**：`src/retrieval/embedder.py`、`src/retrieval/vectorstore.py`

## 1. Embedder 契约
- `BaseEmbedder.embed_documents(texts: list[str]) -> list[list[float]]`
- `BaseEmbedder.embed_query(text: str) -> list[float]`
- 工厂方法 `create_embedder(config)`：
  - `provider == "local"`：使用 `LocalONNXEmbedder`（默认 `Xenova/bge-small-zh-v1.5` 量化 ONNX，L2 归一化）。
  - `provider == "dashscope"`：使用 `DashScopeEmbedder`（需 `.env` 提供 `DASHSCOPE_API_KEY`）。

## 2. 向量库与检索契约
- `RetrievedChunk(chunk_id, doc_id, text, score, metadata)`：
  - `score` 统一归一化为余弦相似度 `[0.0, 1.0]`，数值越高表示越相关，供 `RAGService` 拒答阈值判定。
  - `metadata` 必须保留 `doc_id`、`question_id`、`question`、`source`、`chunk_index`，支撑回答引用溯源。
- `ChromaVectorStore.search(query: str, top_k: int) -> list[RetrievedChunk]`：集合为空或空查询时安全返回空列表 `[]`。
