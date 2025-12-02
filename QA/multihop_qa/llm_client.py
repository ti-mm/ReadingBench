"""文本 LLM 客户端封装，仅支持 OpenAI 兼容接口（可对接 vLLM serve）。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


class BaseLLMClient:
    """统一 ask_json 接口。"""

    def ask_json(self, prompt: str, max_tokens: int = 2048) -> Dict:
        raise NotImplementedError


@dataclass
class ServeLLMConfig:
    model: str = "gpt-3.5-turbo"
    base_url: str = "http://localhost:8000/v1"  # vLLM serve 默认
    api_key: str = "EMPTY"
    temperature: float = 0.0


class ServeLLMClient(BaseLLMClient):
    """调用 vLLM serve（OpenAI 兼容接口）的封装。"""

    def __init__(self, config: ServeLLMConfig):
        self.config = config
        self._client = None

    def _ensure_client(self) -> None:
        if self._client:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 openai 包才能调用 LLM") from exc
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def ask_json(self, prompt: str, max_tokens: int = 2048) -> Dict:
        self._ensure_client()
        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        if not content:
            raise RuntimeError("LLM 返回为空")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM JSON 解析失败: {content}") from exc
