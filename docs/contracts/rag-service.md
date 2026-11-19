# 契约：RAG 核心服务层（`src/rag/`）

- **关联任务**：`T05`（基础 RAG 链路）、`T08`（Query 改写）、`T10`（引用校验与拒答调优）
- **实现文件**：`src/rag/service.py`

## 1. 核心接口
### `RAGService.answer(query: str, history: list[dict] | None = None) -> RAGResponse`
- **输入**：
  - `query`: 用户当前提问。
  - `history`: 标准对话历史列表 `[{"role": "user"|"assistant", "content": "..."}]`。
- **输出**：`RAGResponse`
  - `answer`: 已附加医疗免责声明、且已过滤非法 `[n]` 编号的最终回答文本。
  - `citations`: 回答中实际引用的 `Citation` 列表（`index`, `chunk_id`, `doc_id`, `question`, `text`, `score`, `source`）。
  - `retrieved_chunks`: 本次检索召回的原始 `RetrievedChunk` 列表。
  - `refused`: `bool`，是否触发拒答（前置低相似度拒答或模型 `[REFUSE]` 拒答）。
  - `refusal_reason`: 拒答原因标识（如 `"low_similarity(...)"`、`"llm_refusal"`、`"empty_query"`）。
  - `prompt_tokens` / `completion_tokens` / `total_tokens` / `latency_ms`。

## 2. 行为约定
- 召回最高相似度 `< min_similarity_score` 时直接拒答，不调用 LLM。
- 所有版本必须保留医疗免责声明，不得伪造超出召回数量的引用编号。
