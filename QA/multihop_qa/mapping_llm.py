#!/usr/bin/env python3
"""
针对 LaTeX 包提取并结构化表格内容：
- 找到主 .tex（含 \\documentclass / \\begin{document} 优先）并递归展开其中的 \\input。
- 合并后扫描 \\begin{table...}/\\end{table...} 或 \\begin{tabular...}/\\end{tabular...} 的片段。
- 返回表格全文及前后各 context_chars 个字符的上下文（prefix/suffix）。
- 可选：调用 LLM 将单个表格转为结构化 JSON（caption/columns/rows）。
- 还提供从 parsed_pdfs content_list 过滤正文表格的工具（统计 ref_text 之前的表格数，并只保留正文表格）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from multihop_qa.latex_tables_pylatexenc import (
    body_tables_from_content_list,
    extract_tables_from_package,
    extract_tables_from_text,
)
from multihop_qa.llm_client import BaseLLMClient


def _table_prompt(table_block: str, prefix: str, suffix: str) -> str:
    """
    生成给 LLM 的提示词，将 LaTeX 表格转为结构化 JSON。
    """
    example = (
        "示例输出格式：\n"
        '{\n'
        '  "caption": "Performance comparison on key mathematical reasoning benchmarks.",\n'
        '  "columns": ["Model", "Params", "Tool", "AIME24", "AIME25", "MATH500", "Avg.", "Code Prop."],\n'
        '  "rows": [\n'
        '    {"Model": "Qwen2.5 Ins.", "Params": "7B", "Tool": "✗", "AIME24": "13.3%", "AIME25": "20.0%", "MATH500": "75.8%", "Avg.": "36.4%", "Code Prop.": "0.0"},\n'
        '    {"Model": "ZTRL", "Params": "7B", "Tool": "✓", "AIME24": "46.7%", "AIME25": "30.0%", "MATH500": "85.2%", "Avg.": "54.0%", "Code Prop.": "90%"}\n'
        '  ]\n'
        "}\n"
    )
    rules = (
        "你是 LaTeX 表格解析助手。给定表格片段和前后上下文，输出纯 JSON："
        '{"caption": "...", "columns": ["..."], "rows": [ {col: value, ...}, ...] }。'
        "要求：\n"
        "- caption 优先取表格中的 \\caption 文本，若缺失可用空字符串；\n"
        "- columns 使用表头顺序，去除 LaTeX 控制符；\n"
        "- rows 数量与表格行一致，列名与 columns 对齐；\n"
        "- 去掉格式命令（如 \\textbf、\\underline），将 \\ding{51}/\\checkmark 视为 \"✓\"，\\ding{55} 视为 \"✗\"；\n"
        "- 保留百分号/数字等原始文本，但去除尾部的换行和多余转义；\n"
        "- 只返回 JSON，不要解释文字。"
    )
    parts = [
        rules,
        "表格片段：",
        table_block,
        # "前缀上下文：",
        # prefix,
        # "后缀上下文：",
        # suffix,
        example,
        "请输出 JSON：",
    ]
    return "\n".join(parts)


def table_to_structured_json(
    table_block: str,
    llm_client: BaseLLMClient,
    prefix: str = "",
    suffix: str = "",
    max_tokens: int = 1200,
) -> Dict:
    """
    使用 LLM 将单个 LaTeX 表格转为结构化 JSON（caption/columns/rows）。
    """
    prompt = _table_prompt(table_block, prefix, suffix)
    return llm_client.ask_json(prompt, max_tokens=max_tokens)


def parse_package_tables_with_llm(
    latex_subdir: Path | str,
    llm_client: BaseLLMClient,
    context_chars: int = 500,
    max_tokens: int = 1200,
) -> List[Dict]:
    """
    综合入口：抽取子目录所有表格并用 LLM 解析。
    返回列表，每项包含原始表格字段和 LLM 解析结果，解析失败时会带 error。
    """
    results: List[Dict] = []
    tables = extract_tables_from_package(latex_subdir, context_chars=context_chars)
    for idx, tbl in enumerate(tables):
        item = {
            "table_index": idx,
            "env": tbl.get("env", ""),
            "content": tbl.get("content", ""),
            "prefix": tbl.get("prefix", ""),
            "suffix": tbl.get("suffix", ""),
        }
        try:
            parsed = table_to_structured_json(
                tbl.get("content", ""),
                llm_client=llm_client,
                prefix=tbl.get("prefix", ""),
                suffix=tbl.get("suffix", ""),
                max_tokens=max_tokens,
            )
            item["structured"] = parsed
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
        results.append(item)
    return results
