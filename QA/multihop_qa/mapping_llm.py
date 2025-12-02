#!/usr/bin/env python3
"""
针对 LaTeX 包提取表格内容：
- 找到主 .tex（含 \\documentclass / \\begin{document} 优先）并递归展开其中的 \\input。
- 合并后扫描 \\begin{table...}/\\end{table...} 或 \\begin{tabular...}/\\end{tabular...} 的片段。
- 返回表格全文及前后各 context_chars 个字符的上下文（prefix/suffix）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

INPUT_PATTERN = re.compile(r"\\input\s*(?:\{([^}]+)\}|([^\s%]+))", flags=re.IGNORECASE)
BEGIN_PATTERN = re.compile(r"\\begin\{([^}]+)\}", flags=re.IGNORECASE)


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
    if not Path(cleaned).suffix:
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


def _expand_inputs(text: str, current_dir: Path, root_dir: Path, stack: Set[Path]) -> str:
    """
    递归展开 \\input。stack 用于防止递归循环。
    """

    def replacer(match: re.Match[str]) -> str:
        if _is_commented_line(text, match.start()):
            return match.group(0)

        target = match.group(1) or match.group(2) or ""
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
        expanded = _expand_inputs(content, resolved.parent, root_dir, stack)
        stack.remove(resolved)
        return expanded

    return INPUT_PATTERN.sub(replacer, text)


def find_main_tex(tex_dir: Path) -> Path:
    """
    从目录中选出主 .tex 文件，优先级：
    1) 包含 \\documentclass
    2) 包含 \\begin{document}
    3) 文件名为 main.tex/paper.tex/root.tex
    4) 路径更浅
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
        name_bonus = 1 if path.name.lower() in {"main.tex", "paper.tex", "root.tex"} else 0
        return (has_docclass, has_begin_doc, name_bonus, -rel_depth)

    best = max(tex_files, key=score)
    return best


def merge_latex_dir(tex_dir: Path) -> Tuple[str, Path]:
    """
    返回 (合并后的 tex 文本, 主 tex 路径)。
    """
    tex_dir = tex_dir.resolve()
    main_tex = find_main_tex(tex_dir)
    merged = _expand_inputs(_safe_read_text(main_tex), main_tex.parent, tex_dir, {main_tex})
    return merged, main_tex


def extract_tables_from_text(merged_tex: str, context_chars: int = 500) -> List[Dict[str, str]]:
    """
    从合并后的 tex 中抽取所有表格片段。
    返回元素格式：{"env": "...", "content": "...", "prefix": "...", "suffix": "..."}。
    """
    tables: List[Dict[str, str]] = []
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

        tables.append(
            {
                "env": env,
                "content": merged_tex[start_idx:end_idx],
                "prefix": merged_tex[prefix_start:start_idx],
                "suffix": merged_tex[end_idx:suffix_end],
            }
        )
    return tables


def extract_tables_from_package(latex_subdir: Path | str, context_chars: int = 500) -> List[Dict[str, str]]:
    """
    入口：给定 LaTeX 包子文件夹路径，返回所有表格内容及上下文。
    """
    merged_tex, _ = merge_latex_dir(Path(latex_subdir))
    return extract_tables_from_text(merged_tex, context_chars=context_chars)
