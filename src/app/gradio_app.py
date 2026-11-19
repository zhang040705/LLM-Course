"""MVP 轻量 Gradio Web Demo（薄封装调用 RAGService，业务逻辑全部在 src/rag/）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.config import load_config
from src.rag import RAGResponse, RAGService


def format_citations_markdown(resp: RAGResponse) -> str:
    """将 RAGResponse 中的引用来源与运行状态格式化为 Markdown。"""
    meta_parts = [
        f"**状态**：{'触发拒答 (' + resp.refusal_reason + ')' if resp.refused else '正常作答'}",
        f"**耗时**：`{resp.latency_ms:.0f} ms` ｜ **Token**：`{resp.total_tokens}`",
    ]
    if resp.rewritten_query:
        meta_parts.append(f"**改写检索词**：`{resp.rewritten_query}`")
    if resp.context_stats is not None:
        cs = resp.context_stats
        meta_parts.append(
            f"**上下文预算**：估算 `{cs.estimated_prompt_tokens}` / `{cs.max_context_tokens}` tokens"
            + ("（已按预算裁剪）" if cs.truncated else "")
        )
    if resp.invalid_citations_removed:
        meta_parts.append(f"**已过滤无效引用**：`{resp.invalid_citations_removed}`")
    status_line = "  \n".join(meta_parts)

    if resp.citations:
        lines = [status_line, "", "### 引用来源详情", ""]
        for c in resp.citations:
            lines.append(
                f"- **[{c.index}]** `doc_id={c.doc_id}` ｜ 相似度 `{c.score:.4f}`  \n"
                f"  > {c.text}"
            )
        return "\n".join(lines)

    if resp.retrieved_chunks:
        lines = [
            status_line,
            "",
            "### 检索候选片段（未达引用条件或已拒答）",
            "",
        ]
        for idx, chunk in enumerate(resp.retrieved_chunks[:3], start=1):
            lines.append(
                f"- **候选 {idx}** `doc_id={chunk.doc_id}` ｜ 相似度 `{chunk.score:.4f}`  \n"
                f"  > {chunk.text[:160]}..."
            )
        return "\n".join(lines)

    return status_line + "\n\n暂无检索参考片段。"


def build_gradio_app(
    rag_service: Optional[RAGService] = None,
    config: Optional[Dict[str, Any]] = None,
) -> gr.Blocks:
    """构建 Gradio 多轮医疗问答界面。"""
    cfg = config or load_config()
    disclaimer = cfg.get("rag", {}).get(
        "disclaimer",
        "【医疗免责声明】本系统回答仅供健康科普参考，不构成临床诊疗依据。如有身体不适，请及时前往正规医院就诊。",
    )
    service_holder: Dict[str, Optional[RAGService]] = {"instance": rag_service}

    def _get_service() -> RAGService:
        if service_holder["instance"] is None:
            service_holder["instance"] = RAGService.from_config(cfg)
        return service_holder["instance"]

    def on_submit(
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]],
    ) -> Tuple[str, List[Dict[str, str]], str]:
        history = list(chat_history or [])
        text = (user_message or "").strip()
        if not text:
            return "", history, "请输入具体问题。"

        try:
            service = _get_service()
            resp = service.answer(text, history=history)
            assistant_reply = resp.answer
            citation_md = format_citations_markdown(resp)
        except Exception as exc:
            assistant_reply = f"系统调用异常：{exc}\n\n请检查 `.env` 中的 `DEEPSEEK_API_KEY` 是否已正确配置。"
            citation_md = f"**状态**：调用异常\n\n`{exc}`"

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": assistant_reply})
        return "", history, citation_md

    def on_clear() -> Tuple[str, List[Dict[str, str]], str]:
        return "", [], "对话已清空，请在左侧输入新的医疗健康问题。"

    version = cfg.get("project", {}).get("version", "v1-core")
    with gr.Blocks(title=f"中文医疗领域 RAG 问答系统 ({version})") as demo:
        gr.Markdown(f"# 中文医疗领域特定 RAG 问答系统（{version}）")
        gr.Markdown(f"> {disclaimer}")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="多轮医疗对话", height=460)
                msg_box = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：怀孕25周着凉拉肚子怎么办？",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空历史")

                gr.Examples(
                    examples=[
                        ["怀孕25周着凉拉肚子怎么办？"],
                        ["感冒发烧咳嗽需要注意什么？"],
                        ["Python 中如何使用 asyncio 实现并发请求？"],
                    ],
                    inputs=msg_box,
                    label="快捷测试示例（含域外拒答示例）",
                )

            with gr.Column(scale=2):
                citation_panel = gr.Markdown(
                    value="提问后将在此展示回答引用的原始医疗片段与相似度分数。",
                    label="引用溯源",
                )

        send_btn.click(
            fn=on_submit,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot, citation_panel],
        )
        msg_box.submit(
            fn=on_submit,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot, citation_panel],
        )
        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[msg_box, chatbot, citation_panel],
        )

    return demo
