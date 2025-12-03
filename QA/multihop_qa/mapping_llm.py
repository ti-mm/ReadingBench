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
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from multihop_qa.llm_client import BaseLLMClient

INPUT_PATTERN = re.compile(r"\\input\s*(?:\{([^}]+)\}|([^\s%]+))", flags=re.IGNORECASE)
IMPORT_PATTERN = re.compile(r"\\import\s*\{([^}]*)\}\s*\{([^}]*)\}", flags=re.IGNORECASE)
BEGIN_PATTERN = re.compile(r"\\begin\{([^}]+)\}", flags=re.IGNORECASE)
NEWCOMMAND_PATTERN = re.compile(r"\\newcommand\{\\([^\}]+)\}\{([^}]*)\}")


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _is_commented_line(text: str, index: int) -> bool:
    """
    判断 index 位置是否处于被 % 注释掉的行内。
    """
    line_start = text.rfind("\n", 0, index) + 1
    percent = text.find("%", line_start, index)
    while percent != -1:
        if percent == line_start or text[percent - 1] != "\\":
            return True
        percent = text.find("%", percent + 1, index)
    return False


def _resolve_input_path(target: str, current_dir: Path, root_dir: Path) -> Optional[Path]:
    cleaned = target.strip().strip('"').strip("'")
    if not cleaned:
        return None

    names = [cleaned]
    suffix = Path(cleaned).suffix
    if not suffix or suffix.lower() != ".tex":
        names.append(f"{cleaned}.tex")

    for name in names:
        candidate = Path(name)
        bases = [Path()]  # 用于处理绝对路径的情况
        if not candidate.is_absolute():
            bases = [current_dir, root_dir]
        for base in bases:
            resolved = (base / candidate).resolve()
            if resolved.exists() and resolved.is_file():
                return resolved
    return None


def _resolve_import_path(dir_part: str, file_part: str, current_dir: Path, root_dir: Path) -> Optional[Path]:
    dir_clean = dir_part.strip()
    file_clean = file_part.strip()
    if not file_clean:
        return None

    names = [file_clean]
    suffix = Path(file_clean).suffix
    if not suffix or suffix.lower() != ".tex":
        names.append(f"{file_clean}.tex")

    bases = [Path(dir_clean)] if Path(dir_clean).is_absolute() else [current_dir / dir_clean, root_dir / dir_clean]
    for base in bases:
        for name in names:
            resolved = (base / name).resolve()
            if resolved.exists() and resolved.is_file():
                return resolved
    return None


def _parse_macros(text: str) -> Dict[str, str]:
    """
    解析简单的 \newcommand{\foo}{bar} 宏定义，返回 {\\foo: "bar"}。
    """
    macros: Dict[str, str] = {}
    for m in NEWCOMMAND_PATTERN.finditer(text):
        macros["\\" + m.group(1)] = m.group(2)
    return macros


def _apply_macros(target: str, macros: Dict[str, str]) -> str:
    for name, val in macros.items():
        target = target.replace(name + " ", val)
        target = target.replace(name, val)
    return target


def _expand_inputs(text: str, current_dir: Path, root_dir: Path, stack: Set[Path], macros: Dict[str, str]) -> str:
    """
    递归展开 \\input。stack 用于防止递归循环。
    """
    # 合并当前文本中的宏定义
    local_macros = dict(macros)
    local_macros.update(_parse_macros(text))

    source_for_import = text

    def import_replacer(match: re.Match[str]) -> str:
        if _is_commented_line(source_for_import, match.start()):
            return match.group(0)
        dir_part = _apply_macros(match.group(1) or "", local_macros)
        file_part = _apply_macros(match.group(2) or "", local_macros)
        resolved = _resolve_import_path(dir_part, file_part, current_dir, root_dir)
        if not resolved:
            return ""
        if resolved in stack:
            return ""
        stack.add(resolved)
        try:
            content = _safe_read_text(resolved)
        except Exception:
            stack.remove(resolved)
            return ""
        expanded = _expand_inputs(content, resolved.parent, root_dir, stack, local_macros)
        stack.remove(resolved)
        return expanded

    text_after_import = IMPORT_PATTERN.sub(import_replacer, text)
    source_for_input = text_after_import

    def input_replacer(match: re.Match[str]) -> str:
        if _is_commented_line(source_for_input, match.start()):
            return match.group(0)

        target = _apply_macros(match.group(1) or match.group(2) or "", local_macros)
        resolved = _resolve_input_path(target, current_dir, root_dir)
        if not resolved:
            return ""
        if resolved in stack:
            return ""

        stack.add(resolved)
        try:
            content = _safe_read_text(resolved)
        except Exception:
            stack.remove(resolved)
            return ""
        expanded = _expand_inputs(content, resolved.parent, root_dir, stack, local_macros)
        stack.remove(resolved)
        return expanded

    return INPUT_PATTERN.sub(input_replacer, text_after_import)


def find_main_tex(tex_dir: Path) -> Path:
    """
    从目录中选出主 .tex 文件，优先级：
    1) 包含 \\documentclass
    2) 包含 \\begin{document}
    3) 文件名提示（main/paper/root/arxiv/linearfits 等）
    4) 含 \\input/\\import 越多得分越高
    5) 路径更浅
    """
    tex_files = [p for p in tex_dir.rglob("*.tex") if p.is_file()]
    if not tex_files:
        raise FileNotFoundError(f"未找到 .tex 文件：{tex_dir}")

    def score(path: Path) -> Tuple[int, int, int, int]:
        try:
            txt = _safe_read_text(path).lower()
        except Exception:
            txt = ""
        rel_depth = len(path.relative_to(tex_dir).parts)
        has_docclass = 1 if "\\documentclass" in txt else 0
        has_begin_doc = 1 if "\\begin{document}" in txt else 0
        # 文件名提示
        name = path.name.lower()
        name_hits = ["main", "paper", "root", "arxiv", "linearfits"]
        name_bonus = 1 if any(h in name for h in name_hits) else 0
        # 粗略统计包含的 input/import 数量
        inputs = len(INPUT_PATTERN.findall(txt)) + len(IMPORT_PATTERN.findall(txt))
        return (has_docclass, has_begin_doc, name_bonus, inputs, -rel_depth)

    best = max(tex_files, key=score)
    return best


def merge_latex_dir(tex_dir: Path) -> Tuple[str, Path]:
    """
    返回 (合并后的 tex 文本, 主 tex 路径)。
    """
    tex_dir = tex_dir.resolve()
    main_tex = find_main_tex(tex_dir)
    merged = _expand_inputs(_safe_read_text(main_tex), main_tex.parent, tex_dir, {main_tex}, {})
    return merged, main_tex


def extract_tables_from_text(merged_tex: str, context_chars: int = 500) -> List[Dict[str, str]]:
    """
    从合并后的 tex 中抽取所有表格片段。
    返回元素格式：{"env": "...", "content": "...", "prefix": "...", "suffix": "..."}。
    若存在嵌套（如 table 内含 tabular），仅保留最外层以避免重复。
    """
    tables: List[Dict[str, str]] = []
    spans: List[Dict[str, object]] = []
    for match in BEGIN_PATTERN.finditer(merged_tex):
        if _is_commented_line(merged_tex, match.start()):
            continue
        env = (match.group(1) or "").strip()
        env_lower = env.lower()
        if not (env_lower.startswith("table") or env_lower.startswith("tabular")):
            continue

        end_pattern = re.compile(r"\\end\{" + re.escape(env) + r"\}", flags=re.IGNORECASE)
        end_match = end_pattern.search(merged_tex, match.end())
        if not end_match:
            continue

        start_idx = match.start()
        end_idx = end_match.end()
        prefix_start = max(0, start_idx - context_chars)
        suffix_end = end_idx + context_chars
        spans.append(
            {
                "_start": start_idx,
                "_end": end_idx,
                "env": env,
                "content": merged_tex[start_idx:end_idx],
                "prefix": merged_tex[prefix_start:start_idx],
                "suffix": merged_tex[end_idx:suffix_end],
            }
        )

    # 保留最外层：按起点排序，长度逆序，丢弃被已有 span 完全包含的条目
    kept: List[Dict[str, str]] = []
    for item in sorted(spans, key=lambda x: (x["_start"], -(x["_end"] - x["_start"]))):  # type: ignore[index]
        start_idx = int(item["_start"])  # type: ignore[index]
        end_idx = int(item["_end"])  # type: ignore[index]
        if any(start_idx >= int(k["_start"]) and end_idx <= int(k["_end"]) for k in kept):  # type: ignore[index]
            continue
        kept.append(item)

    for item in kept:
        tables.append(
            {
                "env": item["env"],
                "content": item["content"],
                "prefix": item["prefix"],
                "suffix": item["suffix"],
            }
        )
    return tables


def _load_content_list(arxiv_id: str, base_dir: Path) -> List[Dict]:
    """
    从 parsed_pdfs/{arxiv_id}/vlm/{arxiv_id}_content_list.json 加载内容。
    """
    candidates = [
        base_dir / "test_database" / "parsed_pdfs" / arxiv_id / "vlm" / f"{arxiv_id}_content_list.json",
        base_dir / "parsed_pdfs" / arxiv_id / "vlm" / f"{arxiv_id}_content_list.json",
    ]
    content_path = next((p for p in candidates if p.exists()), None)
    if not content_path:
        return []
    try:
        return json.loads(content_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def body_tables_from_content_list(arxiv_id: str, base_dir: Path) -> Tuple[List[Dict], int]:
    """
    读取 content_list，统计正文表格数（截止条件：ref_text 或 text == 'appendix'），并只保留正文部分的表格条目。
    返回 (正文表格列表, 正文表格数)。
    """
    items = _load_content_list(arxiv_id, base_dir)
    if not items:
        return [], 0

    cutoff_idx: Optional[int] = None
    for idx, item in enumerate(items):
        if item.get("type") == "text":
            text_norm = str(item.get("text", "")).strip().lower()
            if text_norm in {"references", "appendix"}:
                cutoff_idx = idx
                break

    body_tables: List[Dict] = []
    tables_before = 0
    for idx, item in enumerate(items):
        if cutoff_idx is not None and idx >= cutoff_idx:
            break
        if item.get("type") != "table":
            continue
        tables_before += 1
        annotated = dict(item)
        annotated["_orig_idx"] = idx
        body_tables.append(annotated)

    return body_tables, tables_before


def extract_tables_from_package(
    latex_subdir: Path | str,
    context_chars: int = 500,
    limit_by_content_list: bool = False,
    base_dir: Path | str | None = None,
) -> List[Dict[str, str]]:
    """
    入口：给定 LaTeX 包子文件夹路径，返回所有表格内容及上下文。
    若 limit_by_content_list=True，则根据 parsed_pdfs content_list 中 ref_text 前的表格数量，截取前 N 个表格。
    """
    latex_path = Path(latex_subdir)
    merged_tex, _ = merge_latex_dir(latex_path)
    tables = extract_tables_from_text(merged_tex, context_chars=context_chars)

    if limit_by_content_list:
        arxiv_id = latex_path.name
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        _, tables_before = body_tables_from_content_list(arxiv_id, base)
        if tables_before > 0:
            tables = tables[:tables_before]
        print(
            f"[extract_tables_from_package] arxiv_id={arxiv_id}, "
            f"body_tables={tables_before}, extracted_total={len(tables)}"
        )

    return tables


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
