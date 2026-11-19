"""多轮对话 Query 改写模块（对应 T08）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.rag.context_manager import normalize_history

REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手，负责将多轮医疗对话中的最新追问改写为一个独立、完整的检索查询句。

请严格遵守以下规则：
1. 若最新追问包含指代词（如“它”“这个药”“这种情况”）或省略了前文提到的疾病、人群、症状等实体，请结合【近期对话】补全为一句清晰、适合医疗知识库检索的中文问题。
2. 【话题切换保护】如果用户的最新问题明显切换到了与前文医疗背景无关的新话题（例如编程、金融、法律、历史、数学、日常闲聊等），请**直接原样输出最新问题**，绝对不要强行拼接前文的医疗关键词！
3. 如果最新问题本身已经语义完整且无需上下文，请保持原意或略微精简即可。
4. 仅输出改写后的最终查询文本，不要输出解释、前缀或引号。"""


class QueryRewriter:
    """上下文感知的多轮对话 Query 改写器。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient],
        max_history_turns: int = 3,
        max_rewrite_tokens: int = 128,
        disclaimer: str = "",
    ) -> None:
        self.llm_client = llm_client
        self.max_history_turns = max_history_turns
        self.max_rewrite_tokens = max_rewrite_tokens
        self.disclaimer = disclaimer

    def rewrite(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """根据对话历史对当前追问进行指代消解与实体补全。"""
        clean_query = (query or "").strip()
        if not clean_query or self.llm_client is None:
            return clean_query

        norm_history = normalize_history(history, disclaimer=self.disclaimer)
        if not norm_history:
            return clean_query

        recent = norm_history[-(self.max_history_turns * 2) :]
        history_lines = []
        for msg in recent:
            speaker = "用户" if msg["role"] == "user" else "助手"
            # 截断过长历史回复，避免改写调用本身浪费大量 Token
            content = msg["content"][:300]
            history_lines.append(f"{speaker}: {content}")

        user_prompt = (
            "【近期对话】\n"
            + "\n".join(history_lines)
            + f"\n\n【最新问题】\n{clean_query}\n\n请输出改写后的独立查询句："
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = self.llm_client.chat(
                messages,
                temperature=0.0,
                max_tokens=self.max_rewrite_tokens,
            )
            rewritten = resp.content.strip().strip('"\'“”`').splitlines()[0].strip()
            if not rewritten:
                return clean_query
            if len(rewritten) > len(clean_query) * 3 + 80:
                return clean_query
            return rewritten
        except Exception:
            return clean_query
