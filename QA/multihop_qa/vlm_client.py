from __future__ import annotations

import base64
import json
import json_repair
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VLMConfig:
    model: str = "gpt-4o-mini"          # serve 接口使用的模型名
    temperature: float = 0.2
    base_url: Optional[str] = None      # 已有 vLLM serve 的 base_url，如 http://localhost:8000/v1
    api_key: Optional[str] = None       # serve 的 api_key（可为空）
    launch_server: bool = False         # 是否由代码自动启动 vllm serve
    model_path: Optional[str] = None    # 自动启动时的模型路径
    gpus: Optional[List[int]] = None    # 启动 serve 时使用的 GPU，默认前四张 [0,1,2,3]
    port: int = 8000                    # 启动 serve 的端口


class VLMClient:
    """
    简化版 VLM 客户端：
    - 默认连接已有的 OpenAI 兼容服务（base_url + api_key）
    - 可选 launch_server 在代码内启动 vllm serve（占用指定 GPU）
    """

    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._client = None
        self._server_proc: Optional[subprocess.Popen] = None

    def _encode_image(self, image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def _ensure_client(self) -> None:
        if self._client:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("需要安装 openai 包才能调用 VLM") from exc

        base_url = self.config.base_url
        api_key = self.config.api_key or "EMPTY"

        if self.config.launch_server:
            if not self.config.model_path:
                raise RuntimeError("launch_server=True 时需提供 model_path")
            gpus = self.config.gpus or [0, 1, 2, 3]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
            base_url = base_url or f"http://127.0.0.1:{self.config.port}/v1"
            cmd = [
                "vllm",
                "serve",
                self.config.model_path,
                "--port",
                str(self.config.port),
                "--tensor-parallel-size",
                str(len(gpus)),
                "--trust-remote-code",      # 允许运行模型自定义代码
                "--gpu-memory-utilization", "0.95", # 235B 模型很大，最大化显存利用
                "--max-model-len", "8192"   # 防止显存溢出，根据显存大小适当调整
            ]
            self._server_proc = subprocess.Popen(cmd, env=env)
            time.sleep(2)  # 简单等待服务启动

        if not base_url:
            raise RuntimeError("未提供 base_url，且未启用 launch_server")

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _clean_markdown_json(self, content: str) -> str:
        """
        清洗 VLM 输出中可能包含的 Markdown 代码块标记
        """
        content = content.strip()
        # 去掉开头的 ```json 或 ```
        if content.startswith("```"):
            # 找到第一个换行符
            newline_idx = content.find("\n")
            if newline_idx != -1:
                content = content[newline_idx+1:]
            else:
                # 极端情况：只有一行，直接暴力去掉前缀
                content = content.replace("```json", "").replace("```", "")
        
        # 去掉结尾的 ```
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()

    def ask_json(self, image_path: Path, prompt: str, max_tokens: int = 800) -> Dict:
        self._ensure_client()
        assert self._client is not None
        payload = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image_path)}"},
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": payload}],
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        if not content:
            raise RuntimeError("VLM 返回为空")
        
        # === 新增：清洗 Markdown 标记 ===
        cleaned_content = self._clean_markdown_json(content)
        # ==============================

        try:
            return json_repair.loads(cleaned_content)
        except json.JSONDecodeError as exc:
            # 记录原始返回内容以便调试
            raise RuntimeError(f"VLM JSON 解析失败: {content}") from exc

    def read_table(self, image_path: Path) -> Dict:
        prompt = (
            "读取科学论文中的表格图片，返回 JSON，仅包含 rows 数组："
            "{dataset, method, metric_name, metric_value (字符串), rank(可选,1为最优)}。"
            "请保持纯 JSON 响应。"
        )
        return self.ask_json(image_path, prompt, max_tokens=800)


class NullVLMClient(VLMClient):
    def __init__(self):
        super().__init__(config=VLMConfig())

    def read_table(self, image_path: Path) -> Dict:
        raise RuntimeError("未启用 VLM；请提供 serve 配置或启用 launch_server")