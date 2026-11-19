"""OpenAI 兼容大语言模型客户端封装（支持 DeepSeek 云端 API 与本地推理端点）。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


@dataclass
class LLMResponse:
    """大模型单次对话响应结构。"""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient:
    """OpenAI 兼容协议的大语言模型统一客户端。"""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_context_tokens: int = 32768,
        max_output_tokens: int = 1536,
        temperature: float = 0.2,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        resolved_url, resolved_key = self._resolve_credentials(provider, base_url, api_key)
        self.base_url = resolved_url
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """从全局配置字典构建 LLMClient 实例。"""
        llm_cfg = config.get("llm", {})
        return cls(
            provider=llm_cfg.get("provider", "deepseek"),
            model=llm_cfg.get("model", "deepseek-chat"),
            base_url=llm_cfg.get("base_url"),
            max_context_tokens=int(llm_cfg.get("max_context_tokens", 32768)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 1536)),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            timeout=int(llm_cfg.get("timeout", 60)),
        )

    @staticmethod
    def _resolve_credentials(
        provider: str,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> tuple[str, str]:
        if provider == "deepseek":
            url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
            if not key or key == "your_deepseek_api_key_here":
                raise ValueError(
                    "未检测到有效的 DEEPSEEK_API_KEY，请在 .env 文件或环境变量中配置。"
                )
            return url, key

        if provider == "local":
            url = base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
            key = api_key or os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
            return url, key

        raise ValueError(f"不支持的 LLM provider: {provider}（可选: deepseek, local）")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算中英文混合文本的 Token 数（用于上下文预算防溢出保护）。"""
        if not text:
            return 0
        # 中文字符约 1~1.5 token，英文/空白约 0.3~0.5 token，取保守上界
        cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other_count = len(text) - cjk_count
        return int(cjk_count * 1.2 + other_count * 0.4) + 1

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """执行非流式对话补全，内置瞬态网络错误重试。"""
        temp = self.temperature if temperature is None else temperature
        out_tokens = self.max_output_tokens if max_tokens is None else max_tokens

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=out_tokens,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                return LLMResponse(
                    content=content.strip(),
                    model=resp.model or self.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_ms=round(latency_ms, 2),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise
        assert last_err is not None
        raise last_err

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """执行流式对话补全，逐段产出文本增量。"""
        temp = self.temperature if temperature is None else temperature
        out_tokens = self.max_output_tokens if max_tokens is None else max_tokens

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=out_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
