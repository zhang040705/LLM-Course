# 契约：RAG 核心服务层（`src/rag/`）

- **关联任务**：`T05`（基础 RAG 链路）、`T08`（Query 改写）、`T09`（`>32k` 上下文管理）、`T10`（引用校验与拒答调优）
- **实现文件**：`src/rag/service.py`、`src/rag/rewriter.py`、`src/rag/context_manager.py`、`src/rag/citation.py`

## 1. 核心接口
### `RAGService.answer(query: str, history: list[dict] | None = None) -> RAGResponse`
- **输入**：
  - `query`: 用户当前提问。
  - `history`: 标准对话历史列表 `[{"role": "user"|"assistant", "content": "..."}]`（同时兼容 Gradio 多段消息块）。
- **输出**：`RAGResponse`
  - `answer`: 已附加医疗免责声明、且已过滤非法 `[n]` 编号的最终回答文本。
  - `citations`: 回答中实际引用的 `Citation` 列表（`index`, `chunk_id`, `doc_id`, `question`, `text`, `score`, `source`）。
  - `retrieved_chunks`: 经噪声过滤并实际送入上下文的 `RetrievedChunk` 列表。
  - `refused`: `bool`，是否触发拒答（`empty_query` / `low_similarity(...)` / `llm_refusal` / `no_valid_citation`）。
  - `refusal_reason`: 拒答原因标识。
  - `rewritten_query`: 多轮改写后的检索词（单轮或关闭改写时为空字符串 `""`）。
  - `invalid_citations_removed`: 被后置校验剥离的越界/虚构引用编号列表。
  - `context_stats`: `ContextStats`（`estimated_prompt_tokens`, `context_tokens`, `history_tokens`, `included_chunks`, `dropped_chunks`, `included_history_msgs`, `dropped_history_msgs`, `truncated`, `max_context_tokens`）。
  - `prompt_tokens` / `completion_tokens` / `total_tokens` / `latency_ms`。

## 2. 子模块行为约定
- **`QueryRewriter`（`src/rag/rewriter.py`）**：
  - 单轮对话（`history` 为空）零额外 LLM 开销，直接返回原查询。
  - 多轮追问执行指代消解与实体补全；遇到跨领域话题切换时保持原查询，不强行拼接前文医疗实体；异常时安全回退原查询。
- **`ContextBudgetManager`（`src/rag/context_manager.py`）**：
  - 在 `max_context_tokens`（默认 `65536`）预算内优先保障检索证据（Top-1 证据永不静默丢弃），剩余动态预算分配给去除重复免责声明后的近期对话历史。
- **`verify_and_extract_citations`（`src/rag/citation.py`）**：
  - 将 `【1】`、`[1, 2]` 等格式统一规范化为 `[1][2]`，剥离越界编号；当 `require_citation=True` 且非拒答回答完全缺失有效引用时，自动触发 `no_valid_citation` 后置拒答。
- **前置拒答底线**：
  - 召回最高相似度 `< min_similarity_score`（V1 为 `0.52`）时直接拒答，零 LLM Token 消耗；所有版本必须保留医疗免责声明。
