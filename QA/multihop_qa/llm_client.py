"""
文本 LLM 客户端封装，支持：
- serve: OpenAI 兼容接口（如 vLLM serve）
- local: 本地 vLLM (LLM) 纯文本推理
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


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


@dataclass
class LocalLLMConfig:
    model_path: str  # 本地权重路径或 HuggingFace 名称
    gpu_ids: Optional[List[int]] = None  # 默认用后四张卡 [4,5,6,7]
    temperature: float = 0.0

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [4, 5, 6, 7]


class LocalLLMClient(BaseLLMClient):
    """使用 vllm.LLM 在本地推理（纯文本）。"""

    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self._llm = None
        self._sampling_params = None

    def _ensure_client(self) -> None:
        if self._llm:
            return
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 vllm 包才能使用本地 LLM") from exc
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.config.gpu_ids)
        self._llm = LLM(model=self.config.model_path, tensor_parallel_size=len(self.config.gpu_ids))
        self._sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=2048)

    def ask_json(self, prompt: str, max_tokens: int = 2048) -> Dict:
        self._ensure_client()
        params = self._sampling_params
        params.max_tokens = max_tokens
        outputs = self._llm.generate([prompt], params)
        content = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        if not content:
            raise RuntimeError("LLM 返回为空")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM JSON 解析失败: {content}") from exc
