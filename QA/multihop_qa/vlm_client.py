from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    base_url: Optional[str] = None  # 兼容 vLLM serve/openai proxy，例：http://localhost:8000/v1
    api_key: Optional[str] = None   # 若未提供，默认取环境变量或用占位 "EMPTY"


class VLMClient:
    """
    可选的 VLM 调用封装。支持：
    - 官方 OpenAI（默认读取 OPENAI_API_KEY）
    - vLLM serve / 兼容 OpenAI 接口的服务（通过 base_url + api_key 或 VLM_BASE_URL 环境变量）
    """

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._client = None

    def _ensure_client(self) -> None:
        if self._client:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 openai 包才能使用 VLM") from exc
        base_url = self.config.base_url or os.environ.get("VLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLM_API_KEY") or "EMPTY"
        if not base_url and api_key == "EMPTY":
            raise RuntimeError("请设置 OPENAI_API_KEY，或提供 VLM_BASE_URL/VLM_API_KEY 以使用 vLLM serve")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _encode_image(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def read_table(self, image_path: Path) -> Dict[str, Any]:
        """
        通过 VLM 读取表格图片，返回 JSON:
        { rows: [{dataset, method, metric_name, metric_value, rank?}], ranking_hint: ... }
        """
        self._ensure_client()
        assert self._client is not None

        prompt = (
            "读取科学论文中的表格图片，返回 JSON，仅包含 rows 数组："
            "{dataset, method, metric_name, metric_value (字符串), rank(可选,1为最优)}。"
            "请保持纯 JSON 响应。"
        )
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"},
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
        )
        message = resp.choices[0].message.content
        if not message:
            raise RuntimeError("VLM 返回为空")
        try:
            return json.loads(message)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VLM JSON 解析失败: {message}") from exc

    def ask_json(self, image_path: Path, prompt: str, max_tokens: int = 800) -> Dict[str, Any]:
        """
        使用自定义 prompt 读取图片，要求返回 JSON。
        """
        self._ensure_client()
        assert self._client is not None

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"},
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        message = resp.choices[0].message.content
        if not message:
            raise RuntimeError("VLM 返回为空")
        try:
            return json.loads(message)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VLM JSON 解析失败: {message}") from exc


class NullVLMClient(VLMClient):
    """占位符：在没有 VLM 的情况下使用，任何调用都会报错。"""

    def __init__(self):
        super().__init__(config=VLMConfig())

    def read_table(self, image_path: Path) -> Dict[str, Any]:  # type: ignore[override]
        raise RuntimeError("未启用 VLM；请使用 fallback HTML 解析或配置 OPENAI_API_KEY")
