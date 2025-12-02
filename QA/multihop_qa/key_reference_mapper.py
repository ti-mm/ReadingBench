"""
预留接口：将 mapping 结果中的 key（数据集/方法名称）映射到论文引用线索。

当前不实现具体逻辑，供后续开发使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def map_keys_to_references(mapping_result: Dict, contexts_path: Path) -> Dict:
    """
    期望行为（待实现）：
    - 读取 contexts_path（table_text_contexts.jsonl）获取表格周边文本
    - 抽取引用线索并把 mapping_result 中的 key 关联到这些引用
    - 返回包含 key_reference_map 的增强结果

    当前占位：直接抛出 NotImplementedError。
    """
    raise NotImplementedError("map_keys_to_references 仅为占位接口，尚未实现")
