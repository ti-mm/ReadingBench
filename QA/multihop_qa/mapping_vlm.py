#!/usr/bin/env python3
"""
从 MinerU 解析结果中取正文表格，调用 VLM 将表格图片转为结构化 JSON，可选再对照 LaTeX 表格做一致性判定。

示例：
    python QA/multihop_qa/mapping_vlm.py --paper QA/test_database/parsed_pdfs/1610.02136 --verify-with-latex --save-dir /path/to/save
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multihop_qa.models import TableContext
from multihop_qa.latex_tables_regex import extract_tables_from_package
from multihop_qa.vlm_client import VLMClient, VLMConfig

STRUCTURED_TABLE_MAX_TOKENS = 1600
VERIFY_MAX_TOKENS = 1000
STRUCTURED_RETRY = 3
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _trim_text(text: str, limit: int = 3200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _locate_vlm_dir(paper_dir: Path) -> Tuple[Path, Path, Path]:
    """
    返回 (vlm_dir, parsed_root, arxiv_id)。
    paper_dir 可以是 arxiv_id 目录或其下的 vlm 目录。
    """
    paper_dir = paper_dir.resolve()
    if paper_dir.name == "vlm":
        vlm_dir = paper_dir
        arxiv_id = paper_dir.parent.name
        parsed_root = paper_dir.parent.parent
    else:
        candidate = paper_dir / "vlm"
        if candidate.exists():
            vlm_dir = candidate
            arxiv_id = paper_dir.name
            parsed_root = paper_dir.parent
        else:
            vlm_dir = paper_dir
            arxiv_id = paper_dir.stem
            parsed_root = paper_dir.parent
    return vlm_dir, parsed_root, Path(arxiv_id)


def _load_content_list(vlm_dir: Path, arxiv_id: Path) -> List[Dict]:
    candidates = list(vlm_dir.glob("*_content_list.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到 content_list: {vlm_dir}")
    # 优先匹配 arxiv_id
    for c in candidates:
        if arxiv_id.name in c.name:
            return json.loads(c.read_text(encoding="utf-8"))
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _body_only_items(items: List[Dict]) -> List[Dict]:
    """
    截断 content_list：遇到 ref_text 或正文中的 references/appendix 标记即停止。
    """
    cutoff_idx: Optional[int] = None
    for idx, item in enumerate(items):
        if item.get("type") == "ref_text":
            cutoff_idx = idx
            break
        if item.get("type") == "text":
            text_norm = str(item.get("text", "")).strip().lower()
            if text_norm in {"references", "appendix"}:
                cutoff_idx = idx
                break
    return items if cutoff_idx is None else items[:cutoff_idx]


def _contexts_from_items(
    items: List[Dict],
    arxiv_id: str,
    vlm_dir: Path,
    parsed_root: Path,
    window: int = 10,
) -> List[TableContext]:
    text_positions: List[int] = []
    text_values: List[str] = []
    for idx, item in enumerate(items):
        if item.get("type") == "text":
            text_positions.append(idx)
            text_values.append(item.get("text", ""))

    contexts: List[TableContext] = []
    for idx, item in enumerate(items):
        if item.get("type") != "table":
            continue
        # 找到前/后 window 个 text
        insertion = 0
        while insertion < len(text_positions) and text_positions[insertion] < idx:
            insertion += 1
        start = max(0, insertion - window)
        end = insertion + window
        before = text_values[start:insertion]
        after = text_values[insertion:end]

        img_path = item.get("img_path")
        rel_image = str((vlm_dir / img_path).relative_to(parsed_root)) if img_path else ""

        contexts.append(
            TableContext(
                arxiv_id=arxiv_id,
                page_idx=item.get("page_idx", -1),
                table_entry_index=idx,
                image_path=rel_image,
                table_caption=item.get("table_caption", []),
                table_footnote=item.get("table_footnote", []),
                table_body=item.get("table_body", ""),
                text_before=before,
                text_after=after,
            )
        )
    return contexts


def _structured_table_prompt(caption: List[str]) -> str:
    cap = " ".join(caption).strip()
    cap_line = f"表格 caption: {cap}" if cap else "表格 caption: （未提供）"
    return (
        "你将收到一张表格图片，请将其转换为结构化 JSON。表格可能包含以下结构：\n\n"
        "- 可能有 group（大分组）\n"
        "- 可能有 subgroup（小分组）\n"
        "- 可能没有分组\n\n"
        "- 列头可能有多级结构：column → subcolumn → subsubcolumn，最多不超过三级。\n"
        "- 行数据与列结构对应。\n\n"
        "请按以下规则输出 JSON：\n"
        "===============\n【1. 分组结构】\n===============\n\n"
        "若表格包含分组：\n\n"
        "{\n"
        '  "caption": "...",\n'
        '  "groups": {\n'
        '    "GroupName": {\n'
        '      "subgroups": {\n'
        '        "SubGroupName": {\n'
        '          "columns": [...],\n'
        '          "rows": [...]\n'
        "        }\n"
        "      },\n"
        '      "columns": [...],\n'
        '      "rows": [...]\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "说明：group 是第一层，subgroup 是第二层；无分组时不要输出 groups 字段，直接给 columns/rows。\n\n"
        "===============\n【2. 列结构最多三层】\n===============\n\n"
        "列结构规则：\n"
        "(1) 只有一层列头：\n"
        '\"columns\": [\"Lv.1\", \"Lv.2\", \"Avg.\"]\n\n'
        "(2) 两层列头：\n"
        '\"columns\": [\n'
        '  {\n'
        '    \"column\": \"General AI Assistant\",\n'
        '    \"subcolumns\": [\"Lv.1\", \"Lv.2\", \"Lv.3\", \"Avg.\"]\n'
        "  }\n"
        "]\n\n"
        "(3) 三层列头：\n"
        '\"columns\": [\n'
        '  {\n'
        '    \"column\": \"Humanity’s Last Exam\",\n'
        '    \"subcolumns\": [\n'
        '      {\n'
        '        \"subcolumn\": \"Knowledge\",\n'
        '        \"subsubcolumns\": [\"NS\", \"CE\", \"SF\"]\n'
        "      },\n"
        '      \"Avg.\"\n'
        "    ]\n"
        "  }\n"
        "]\n\n"
        "超过三层时，将多余层级平铺到 subsubcolumns。\n\n"
        "===============\n【3. 行结构对齐列结构】\n===============\n"
        "行数据必须与列结构对应，例如：\n"
        "{\n"
        '  \"Method\": \"...\",\n'
        '  \"General AI Assistant\": {\n'
        '    \"Lv.1\": \"...\",\n'
        '    \"Lv.2\": \"...\",\n'
        '    \"Lv.3\": \"...\",\n'
        '    \"Avg.\": \"...\"\n'
        "  },\n"
        '  \"Humanity’s Last Exam\": {\n'
        '    \"Knowledge\": {\n'
        '      \"NS\": \"...\",\n'
        '      \"CE\": \"...\",\n'
        '      \"SF\": \"...\"\n'
        "    },\n"
        '    \"Avg.\": \"...\"\n'
        "  }\n"
        "}\n\n"
        "===============\n【4. 数据格式要求】\n===============\n"
        "- 所有数值都输出字符串；\n"
        "- 空白/破折号用 null；\n"
        "- 列名中的脚注符号（如 †、*）必须保留；\n"
        "- 不输出 Markdown、解释或代码块，只能返回 JSON。\n\n"
        "===============\n【5. 顶层结构】\n===============\n"
        "无分组：{\n"
        '  \"caption\": \"...\",\n'
        '  \"columns\": [...],\n'
        '  \"rows\": [...]\n'
        "}\n\n"
        "有分组：{\n"
        '  \"caption\": \"...\",\n'
        '  \"groups\": { ... }\n'
        "}\n\n"
        f"{cap_line}\n"
        "重要格式要求：\n"
        "1. 请直接返回纯 JSON 字符串。\n"
        "2. 严禁使用 Markdown 代码块格式（即不要用 ```json 包裹）。\n"
        "3. 不要输出任何解释性文字，只返回 JSON 数据。"
    )


def _structured_table_from_image(ctx: TableContext, parsed_root: Path, vlm: VLMClient) -> Dict:
    prompt = _structured_table_prompt(ctx.table_caption)
    last_exc: Exception | None = None
    for _ in range(STRUCTURED_RETRY):
        try:
            return vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=STRUCTURED_TABLE_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"structured table parse failed after {STRUCTURED_RETRY} retries: {last_exc}")


def _verify_prompt(structured_json: Dict, latex_table: Dict) -> str:
    latex_body = latex_table.get("content") or ""
    latex_text = _trim_text(latex_body, limit=3000)
    structured_text = json.dumps(structured_json, ensure_ascii=False)
    # 调整 JSON 输出顺序：先 analysis (Reasoning)，后 conclusions (Tag)
    return (
        "你会看到表格图片（已附在消息中）、该表格的 LaTeX 片段，以及从图片抽取的 JSON。"
        "任务：\n"
        "1) 仔细逐项对比数值和结构，判断 JSON 是否对应这段 LaTeX 的同一张表。\n"
        "2) 先进行详细的思维链分析（analysis），指出匹配点或差异点。\n"
        "3) 最后给出判断结论（same_table, content_match）。\n\n"
        "请严格遵守以下 JSON 格式顺序返回（**先分析，后结论**）：\n"
        "{\n"
        '  "analysis": "在此处进行详细的对比分析，检查行、列、数值是否一致...",\n'
        '  "problems": ["如果不一致，列出具体差异...", "如果一致，请留空"],\n'
        '  "same_table": true/false,\n'
        '  "content_match": true/false\n'
        "}\n\n"
        "重要格式警告：\n"
        "- 必须先输出 analysis 字段，再输出结论。\n"
        "- 必须直接返回纯 JSON 字符串。\n"
        "- 绝对不要使用 Markdown 代码块（不要用 ```json ... ```）。\n"
        "- 不要包含任何其他前缀或后缀文字。\n"
        f"\n\nLaTeX 表格片段（忽略 caption）：\n{latex_text}\n\n图片抽取 JSON：\n{structured_text}"
    )


def _verify_structured_with_latex(
    ctx: TableContext, parsed_root: Path, vlm: VLMClient, structured_json: Dict, latex_table: Dict
) -> Dict:
    prompt = _verify_prompt(structured_json, latex_table)
    return vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=VERIFY_MAX_TOKENS)


def _extract_numbers_from_obj(obj: object) -> Set[str]:
    """
    递归抽取对象中出现的数字字符串，返回去重后的集合。
    """
    nums: Set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            nums |= _extract_numbers_from_obj(v)
    elif isinstance(obj, list):
        for v in obj:
            nums |= _extract_numbers_from_obj(v)
    elif isinstance(obj, (int, float)):
        nums.add(str(obj))
    elif isinstance(obj, str):
        for match in NUMBER_PATTERN.findall(obj):
            cleaned = match.replace(",", "")
            if cleaned:
                nums.add(cleaned)
    return nums


def _numbers_from_structured_table(structured_json: Dict) -> Set[str]:
    return _extract_numbers_from_obj(structured_json)


def _numbers_from_latex_table(latex_table: Dict) -> Set[str]:
    return _extract_numbers_from_obj(latex_table.get("content", ""))


def _numeric_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _best_latex_match(
    structured_json: Dict,
    latex_tables: List[Dict],
    latex_numbers_cache: Optional[List[Set[str]]] = None,
) -> Tuple[Optional[int], float]:
    """
    基于数字的 Jaccard 相似度，从 latex_tables 中挑选与 structured_json 最相似的表格。
    返回 (best_index, similarity_score)，若 latex_tables 为空则返回 (None, 0.0)。
    """
    if not latex_tables:
        return None, 0.0
    structured_nums = _numbers_from_structured_table(structured_json)
    best_idx: Optional[int] = None
    best_score = -1.0
    for idx, latex_tbl in enumerate(latex_tables):
        latex_nums = latex_numbers_cache[idx] if latex_numbers_cache and idx < len(latex_numbers_cache) else _numbers_from_latex_table(latex_tbl)
        score = _numeric_similarity(structured_nums, latex_nums)
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is None:
        return None, 0.0
    return best_idx, best_score


def _load_latex_tables(arxiv_id: Path, base_dir: Path) -> List[Dict]:
    candidates = [
        base_dir / "test_database" / "latex_src" / arxiv_id.name,
        base_dir / "latex_src" / arxiv_id.name,
    ]
    latex_dir = next((p for p in candidates if p.exists()), None)
    if not latex_dir:
        print(f"[Warning] Latex directory not found in candidates: {candidates}")
        return []
    try:
        tables = extract_tables_from_package(
            latex_dir, context_chars=400, limit_by_content_list=True, base_dir=base_dir
        )
        print(f"[Info] Found {len(tables)} tables in Latex source.")
        return tables
    except Exception as e:
        print(f"[Error] Failed to extract latex tables: {e}")
        return []


def _make_vlm_client(
    model: str,
    base_url: str | None,
    api_key: str | None,
    launch_server: bool,
    model_path: str | None,
    gpus: Optional[List[int]],
    port: int,
) -> VLMClient:
    return VLMClient(
        config=VLMConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            launch_server=launch_server,
            model_path=model_path,
            gpus=gpus or [0, 1, 2, 3],
            port=port,
        )
    )


def build_paper_hops(
    paper_dir: Path,
    extract_vlm_model: str = "gpt-4o-mini",
    extract_vlm_base_url: str | None = None,
    extract_vlm_api_key: str | None = None,
    extract_vlm_launch_server: bool = False,
    extract_vlm_model_path: str | None = None,
    extract_vlm_gpus: str | None = None,  # 逗号分隔，如 "0,1,2,3"
    extract_vlm_port: int = 8000,
    verify_vlm_model: str | None = None,
    verify_vlm_base_url: str | None = None,
    verify_vlm_api_key: str | None = None,
    verify_vlm_launch_server: bool | None = None,
    verify_vlm_model_path: str | None = None,
    verify_vlm_gpus: str | None = None,
    verify_vlm_port: int | None = None,
    window: int = 10,
    body_only: bool = True,
    verify_with_latex: bool = False,
    latex_base: Path | None = None,
    save_dir: Path | None = None,
) -> Dict:
    """
    返回结构:
    {
      "paper_id": arxiv_id,
      "tables": [
         {
           "page_idx": ...,
           "table_entry_index": ...,
           "image_path": "...",
           "structured": {...},
           "latex_verification": {...},  # 可选
           "matched_latex_content": "..." # 可选，调试用
         }, ...
      ]
    }
    """
    vlm_dir, parsed_root, arxiv_id = _locate_vlm_dir(paper_dir)
    items = _load_content_list(vlm_dir, arxiv_id)
    if body_only:
        items = _body_only_items(items)
    contexts = _contexts_from_items(items, arxiv_id.name, vlm_dir, parsed_root, window=window)

    latex_tables: List[Dict] = []
    latex_numbers: List[Set[str]] = []
    if verify_with_latex:
        latex_tables = _load_latex_tables(arxiv_id, latex_base or PACKAGE_ROOT)
        latex_numbers = [_numbers_from_latex_table(tbl) for tbl in latex_tables]

    extract_gpu_ids = [int(x) for x in extract_vlm_gpus.split(",") if x.strip().isdigit()] if extract_vlm_gpus else None
    vlm_client = _make_vlm_client(
        model=extract_vlm_model,
        base_url=extract_vlm_base_url,
        api_key=extract_vlm_api_key,
        launch_server=extract_vlm_launch_server,
        model_path=extract_vlm_model_path,
        gpus=extract_gpu_ids,
        port=extract_vlm_port,
    )

    # --- 自定义保存目录设置 ---
    if save_dir:
        BASE_SAVE_DIR = save_dir
    else:
        # 默认回退路径
        BASE_SAVE_DIR = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/multihop_qa/Table_Json")
    
    # 为当前论文创建一个专属子文件夹: {BASE_SAVE_DIR}/{arxiv_id}/
    PAPER_SAVE_DIR = BASE_SAVE_DIR / arxiv_id.name
    PAPER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # -----------------------

    verify_gpu_ids = extract_gpu_ids
    if verify_vlm_gpus:
        verify_gpu_ids = [int(x) for x in verify_vlm_gpus.split(",") if x.strip().isdigit()]
    verify_client = vlm_client
    if verify_with_latex:
        verify_client = _make_vlm_client(
            model=verify_vlm_model or extract_vlm_model,
            base_url=verify_vlm_base_url if verify_vlm_base_url is not None else extract_vlm_base_url,
            api_key=verify_vlm_api_key if verify_vlm_api_key is not None else extract_vlm_api_key,
            launch_server=verify_vlm_launch_server if verify_vlm_launch_server is not None else extract_vlm_launch_server,
            model_path=verify_vlm_model_path if verify_vlm_model_path is not None else extract_vlm_model_path,
            gpus=verify_gpu_ids,
            port=verify_vlm_port if verify_vlm_port is not None else extract_vlm_port,
        )

    tables = []
    for t_idx, ctx in enumerate(contexts):
        record: Dict[str, object] = {
            "page_idx": ctx.page_idx,
            "table_entry_index": ctx.table_entry_index,
            "table_order_index": t_idx,
            "image_path": str(ctx.image_full_path(parsed_root)),
        }
        
        # 1. 提取结构化数据
        try:
            record["structured"] = _structured_table_from_image(ctx, parsed_root, vlm_client)
        except Exception as exc:  # noqa: BLE001
            record["structured_error"] = str(exc)

        # 2. 验证 (如果有 Latex)
        if verify_with_latex:
            if not latex_tables:
                record["latex_verification"] = {"error": "latex_tables_not_found"}
            elif "structured" not in record:
                record["latex_verification"] = {"error": "structured_table_missing"}
            else:
                best_idx, best_score = _best_latex_match(record["structured"], latex_tables, latex_numbers)  # type: ignore[arg-type]
                record["latex_match_index"] = best_idx
                record["latex_match_score"] = best_score
                
                # 将匹配到的 Latex 源码存入 JSON
                if best_idx is not None:
                    record["matched_latex_content"] = latex_tables[best_idx].get("content", "")
                else:
                    record["matched_latex_content"] = None
                
                if best_idx is None:
                    record["latex_verification"] = {"error": "latex_table_not_aligned"}
                else:
                    latex_tbl = latex_tables[best_idx]
                    try:
                        record["latex_verification"] = _verify_structured_with_latex(
                            ctx,
                            parsed_root=parsed_root,
                            vlm=verify_client,
                            structured_json=record["structured"],  # type: ignore[arg-type]
                            latex_table=latex_tbl,
                        )
                    except Exception as exc:  # noqa: BLE001
                        record["latex_verification"] = {"error": str(exc)}

        # 3. 将包含验证信息的完整 record 写入文件
        try:
            # 命名格式: {pdf名字}_table_{第几个table}.json
            file_name = f"{arxiv_id.name}_table_{t_idx}.json"
            save_path = PAPER_SAVE_DIR / file_name
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"[Saved] {save_path} (Included Latex Verification: {'latex_verification' in record})")
        except Exception as e:
            print(f"[Error] Failed to save file: {e}")

        tables.append(record)
    return {"paper_id": arxiv_id.name, "tables": tables}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为单篇论文构建基于表格的多跳 hop 结构（依赖 VLM）")
    parser.add_argument("--paper", type=Path, required=True, help="mineru 处理后的论文目录（arxiv_id 或其下的 vlm）")
    # Extract VLM
    parser.add_argument("--extract-vlm-model", type=str, default="gpt-4o-mini", help="表格抽取阶段的 VLM 模型名")
    parser.add_argument("--extract-vlm-base-url", type=str, default="http://localhost:8000/v1", help="表格抽取 VLM 的 base_url（如 http://localhost:8000/v1）")
    parser.add_argument("--extract-vlm-api-key", type=str, default=None, help="表格抽取 VLM 的 api_key")
    parser.add_argument("--extract-vlm-launch-server", action="store_true", help="是否在本进程内启动表格抽取 VLM（vllm serve）")
    parser.add_argument("--extract-vlm-model-path", type=str, default=None, help="抽取 VLM 启动时的模型路径")
    parser.add_argument("--extract-vlm-gpus", type=str, default=None, help="抽取 VLM 使用的 GPU，逗号分隔，默认前四张 0,1,2,3")
    parser.add_argument("--extract-vlm-port", type=int, default=8000, help="抽取 VLM 的服务端口")
    parser.add_argument("--window", type=int, default=10, help="表格前后收集的 text 数量")
    # Verification VLM overrides
    parser.add_argument("--verify-vlm-model", type=str, default=None, help="验证阶段使用的 VLM 模型名（默认与抽取一致）")
    parser.add_argument("--verify-vlm-base-url", type=str, default=None, help="验证阶段 VLM base_url（默认与抽取一致）")
    parser.add_argument("--verify-vlm-api-key", type=str, default=None, help="验证阶段 VLM api_key（默认与抽取一致）")
    parser.add_argument("--verify-vlm-launch-server", action="store_true", help="验证阶段是否在本进程内启动 vllm serve")
    parser.add_argument("--verify-vlm-model-path", type=str, default=None, help="验证阶段启动 vllm serve 时的模型路径")
    parser.add_argument("--verify-vlm-gpus", type=str, default=None, help="验证阶段使用的 GPU，逗号分隔")
    parser.add_argument("--verify-vlm-port", type=int, default=None, help="验证阶段启动 vllm serve 的端口")
    parser.add_argument(
        "--body-only",
        action="store_true",
        default=True,
        help="仅使用正文表格（遇到 references/appendix/ref_text 即停止），默认开启",
    )
    parser.add_argument(
        "--all-tables",
        action="store_false",
        dest="body_only",
        help="禁用正文截断，使用全部表格（不推荐）",
    )
    parser.add_argument(
        "--verify-with-latex",
        action="store_true",
        help="在 structured 模式下，对照 LaTeX 表格让 VLM 进行一致性判定",
    )
    parser.add_argument(
        "--latex-base",
        type=Path,
        default=None,
        help="LaTeX 源码所在根目录（默认尝试 test_database/latex_src/{arxiv_id}）",
    )
    # 新增参数
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="结果保存的根目录，脚本会自动在其中创建 {arxiv_id} 子目录。如果不传，则使用代码内置的默认路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hops = build_paper_hops(
        paper_dir=args.paper,
        extract_vlm_model=args.extract_vlm_model,
        extract_vlm_base_url=args.extract_vlm_base_url,
        extract_vlm_api_key=args.extract_vlm_api_key,
        extract_vlm_launch_server=args.extract_vlm_launch_server,
        extract_vlm_model_path=args.extract_vlm_model_path,
        extract_vlm_gpus=args.extract_vlm_gpus,
        extract_vlm_port=args.extract_vlm_port,
        verify_vlm_model=args.verify_vlm_model,
        verify_vlm_base_url=args.verify_vlm_base_url,
        verify_vlm_api_key=args.verify_vlm_api_key,
        verify_vlm_launch_server=args.verify_vlm_launch_server if args.verify_vlm_launch_server else None,
        verify_vlm_model_path=args.verify_vlm_model_path,
        verify_vlm_gpus=args.verify_vlm_gpus,
        verify_vlm_port=args.verify_vlm_port,
        window=args.window,
        body_only=args.body_only,
        verify_with_latex=args.verify_with_latex,
        latex_base=args.latex_base,
        save_dir=args.save_dir,
    )
    print(json.dumps(hops, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()