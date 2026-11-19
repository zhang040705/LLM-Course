# 契约：LLM 接入层（`src/llm/`）

- **关联任务**：`T04`（MVP DeepSeek 接入）、`T13`（本地开源模型无缝切换）
- **实现文件**：`src/llm/client.py`

## 1. 接口定义

### `LLMClient.from_config(config: dict) -> LLMClient`
读取 `config["llm"]` 与环境变量构建客户端：
- `provider == "deepseek"`：读取 `DEEPSEEK_API_KEY` 与 `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com`）。
- `provider == "local"`：读取 `LOCAL_LLM_API_KEY`（默认 `EMPTY`）与 `LOCAL_LLM_BASE_URL`（默认 `http://localhost:8000/v1`）。

### `LLMClient.chat(messages, temperature=None, max_tokens=None) -> LLMResponse`
- **输入**：标准 OpenAI `messages` 列表（`[{"role": "...", "content": "..."}]`）。
- **输出**：`LLMResponse(content, model, prompt_tokens, completion_tokens, total_tokens, latency_ms)`。

### `LLMClient.stream_chat(messages, temperature=None, max_tokens=None) -> Iterator[str]`
- 逐块产出文本增量，供后续 FastAPI SSE 与 Gradio 流式输出使用。

## 2. 错误与边界语义
- 缺少有效 `DEEPSEEK_API_KEY` 时在初始化阶段即抛出 `ValueError`，不延迟到请求期才静默失败。
- 对 `APIConnectionError`、`APITimeoutError`、`RateLimitError` 内置最多 3 次指数退避重试；耗尽后向上抛出原异常。
