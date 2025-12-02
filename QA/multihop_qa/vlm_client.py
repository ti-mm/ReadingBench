from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VLMConfig:
    model: str = "gpt-4o-mini"               # serve 模式的模型名称
    temperature: float = 0.2
    base_url: Optional[str] = None           # serve 模式 base_url
    api_key: Optional[str] = None            # serve 模式 api_key
    mode: str = "serve"                      # "serve" (OpenAI 兼容) 或 "local" (本地 vLLM serve)
    model_path: Optional[str] = None         # local 模式下 vLLM 模型路径（可选）
    gpu_ids: Optional[list[int]] = None      # local 模式 GPU 列表，默认用前四张卡 [0,1,2,3]


class VLMClient:
    """
    VLM 调用封装。参考 vLLM 多模态接口（https://docs.vllm.com.cn/en/latest/serving/multimodal_inputs.html）。
    - serve: 直接调用 OpenAI 兼容接口（含 vLLM serve）
    - local: 假定本地已启动 vLLM serve；默认 base_url=http://localhost:8000/v1，默认 GPU [0,1,2,3]
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

        if self.config.mode == "local":
            # 本地 vLLM serve：默认走 localhost，允许通过 gpu_ids 限制可见卡（前四张卡）。
            gpu_ids = self.config.gpu_ids or [0, 1, 2, 3]
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in gpu_ids))
            base_url = self.config.base_url or "http://localhost:8000/v1"
            api_key = self.config.api_key or "EMPTY"
        else:
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
        prompt = (
            "读取科学论文中的表格图片，返回 JSON，仅包含 rows 数组："
            "{dataset, method, metric_name, metric_value (字符串), rank(可选,1为最优)}。"
            "请保持纯 JSON 响应。"
        )
        return self.ask_json(image_path, prompt, max_tokens=800)

    def ask_json(self, image_path: Path, prompt: str, max_tokens: int = 800) -> Dict[str, Any]:
        """
        使用自定义 prompt 读取图片，要求返回 JSON。
        """
        self._ensure_client()
        assert self._client is not None
        content_payload = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"},
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": content_payload}],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""

        if not content:
            raise RuntimeError("VLM 返回为空")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VLM JSON 解析失败: {content}") from exc


class NullVLMClient(VLMClient):
    """占位符：在没有 VLM 的情况下使用，任何调用都会报错。"""

    def __init__(self):
        super().__init__(config=VLMConfig())

    def read_table(self, image_path: Path) -> Dict[str, Any]:  # type: ignore[override]
        raise RuntimeError("未启用 VLM；请使用 fallback HTML 解析或配置 OPENAI_API_KEY")
