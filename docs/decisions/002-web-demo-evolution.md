# ADR-002：Web Demo 演进路线（Gradio → FastAPI）

- **状态**：已采纳
- **背景**：课程要求部署支持实时交互的 Web Demo（Gradio / Streamlit / FastAPI 均可），项目希望 MVP 阶段最快跑通，后期再提升工程化与接口扩展能力。

## 备选方案
- A. 起步直接编写 FastAPI + 前后端联调（初期样板代码多，拖慢 MVP 验证）。
- B. MVP 使用最轻量的 **Gradio** 快速交付对话界面，后期升级为 **FastAPI**（REST + SSE 流式接口）。

## 决策与理由
- **决策**：采用 **方案 B**。
- **架构前提**：所有 RAG 业务逻辑必须下沉至 `src/rag/` 服务层，`src/app/gradio_app.py` 与后期的 `src/app/fastapi_app.py` 均仅作为薄壳调用 `RAGService`，确保迁移 FastAPI 时核心代码零重写。
