# 契约：FastAPI Web 服务接口（`src/app/fastapi_app.py`）

- **关联任务**：`T15`（FastAPI 服务端与交互页面升级）
- **实现文件**：`src/app/fastapi_app.py`、`src/app/static/index.html`、`src/app/static/styles.css`、`src/app/static/mri_shader.js`、`src/app/static/app.js`

## 1. HTTP 接口清单

| 路径 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/` | `GET` | 返回「全息核磁切片台」单页前端（`src/app/static/index.html`） |
| `/static/*` | `GET` | 静态资源挂载（CSS、WebGL Shader JS、前端交互 JS、核磁底纹纹理） |
| `/docs` | `GET` | FastAPI 自动生成的 OpenAPI Swagger 文档 |
| `/api/health` | `GET` | 返回服务状态、版本、检索模式、重排开关、`max_context_tokens`、`min_similarity_score` 与医疗免责声明 |
| `/api/chat` | `POST` | 非流式多轮问答，返回完整结构化 `RAGResponse` JSON（含 `max_score`） |
| `/api/chat/stream` | `POST` | SSE（`text/event-stream`）流式问答接口 |

## 2. 请求与响应结构

### 请求体（`/api/chat` & `/api/chat/stream`）
- `query`: 非空字符串
- `history`: `[{"role": "user"|"assistant", "content": "..."}]`（可选，默认 `[]`）

### SSE 事件协议（`/api/chat/stream`）
每条事件格式为 `data: <JSON>\n\n`：
1. `{"type": "meta", "rewritten_query": "...", "retrieved_count": 5}`
2. `{"type": "delta", "content": "..."}`（多条文本增量）
3. `{"type": "done", "response": {answer, citations, retrieved_count, max_score, refused, refusal_reason, rewritten_query, latency_ms, context_stats, ...}}`
4. 异常时返回 `{"type": "error", "message": "..."}`
