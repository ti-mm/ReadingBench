#!/usr/bin/env python3
"""
从 MinerU 解析结果中取正文表格，调用 VLM 将表格图片转为结构化 JSON，可选再对照 LaTeX 表格做一致性判定。

示例：
    python QA/multihop_qa/mapping_vlm.py --paper QA/test_database/parsed_pdfs/1610.02136 --verify-with-latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multihop_qa.models import TableContext
from multihop_qa.mapping_llm import extract_tables_from_package
from multihop_qa.vlm_client import VLMClient, VLMConfig

STRUCTURED_TABLE_MAX_TOKENS = 1600
VERIFY_MAX_TOKENS = 600


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
        "只返回 JSON。"
    )


def _structured_table_from_image(ctx: TableContext, parsed_root: Path, vlm: VLMClient) -> Dict:
    prompt = _structured_table_prompt(ctx.table_caption)
    return vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=STRUCTURED_TABLE_MAX_TOKENS)


def _verify_prompt(structured_json: Dict, latex_table: Dict) -> str:
    latex_body = latex_table.get("content") or ""
    latex_text = _trim_text(latex_body, limit=3000)
    structured_text = json.dumps(structured_json, ensure_ascii=False)
    return (
        "你会看到表格图片（已附在消息中）、该表格的 LaTeX 片段，以及从图片抽取的 JSON。"
        "任务：\n"
        "1) 判断 JSON 是否对应这段 LaTeX 的同一张表（same_table）。\n"
        "2) 若对应，检查列/分组/数据是否一致，指出缺失或错误。\n"
        '仅返回 JSON：{"same_table": true/false, "content_match": true/false, "problems": ["..."], "summary": "..."}。'
        "如果不是同一张表，content_match 必须为 false，并在 problems 写原因。"
        f"\n\nLaTeX 表格片段：\n{latex_text}\n\n图片抽取 JSON：\n{structured_text}"
    )


def _verify_structured_with_latex(
    ctx: TableContext, parsed_root: Path, vlm: VLMClient, structured_json: Dict, latex_table: Dict
) -> Dict:
    prompt = _verify_prompt(structured_json, latex_table)
    return vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=VERIFY_MAX_TOKENS)


def _load_latex_tables(arxiv_id: Path, base_dir: Path) -> List[Dict]:
    candidates = [
        base_dir / "test_database" / "latex_src" / arxiv_id.name,
        base_dir / "latex_src" / arxiv_id.name,
    ]
    latex_dir = next((p for p in candidates if p.exists()), None)
    if not latex_dir:
        return []
    try:
        return extract_tables_from_package(
            latex_dir, context_chars=400, limit_by_content_list=True, base_dir=base_dir
        )
    except Exception:
        return []


def build_paper_hops(
    paper_dir: Path,
    vlm_model: str = "gpt-4o-mini",
    vlm_base_url: str | None = None,
    vlm_api_key: str | None = None,
    vlm_launch_server: bool = False,
    vlm_model_path: str | None = None,
    vlm_gpus: str | None = None,  # 逗号分隔，如 "0,1,2,3"
    vlm_port: int = 8000,
    window: int = 10,
    body_only: bool = True,
    verify_with_latex: bool = False,
    latex_base: Path | None = None,
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
           "latex_verification": {...}  # 可选
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
    if verify_with_latex:
        latex_tables = _load_latex_tables(arxiv_id, latex_base or PACKAGE_ROOT)

    gpu_ids = None
    if vlm_gpus:
        gpu_ids = [int(x) for x in vlm_gpus.split(",") if x.strip().isdigit()]
    vlm_client = VLMClient(
        config=VLMConfig(
            model=vlm_model,
            base_url=vlm_base_url,
            api_key=vlm_api_key,
            launch_server=vlm_launch_server,
            model_path=vlm_model_path,
            gpus=gpu_ids or [0, 1, 2, 3],
            port=vlm_port,
        )
    )

    tables = []
    for t_idx, ctx in enumerate(contexts):
        record: Dict[str, object] = {
            "page_idx": ctx.page_idx,
            "table_entry_index": ctx.table_entry_index,
            "table_order_index": t_idx,
            "image_path": str(ctx.image_full_path(parsed_root)),
        }
        try:
            record["structured"] = _structured_table_from_image(ctx, parsed_root, vlm_client)
        except Exception as exc:  # noqa: BLE001
            record["structured_error"] = str(exc)

        if verify_with_latex:
            if not latex_tables:
                record["latex_verification"] = {"error": "latex_tables_not_found"}
            elif "structured" not in record:
                record["latex_verification"] = {"error": "structured_table_missing"}
            elif t_idx >= len(latex_tables):
                record["latex_verification"] = {"error": "latex_table_not_aligned"}
            else:
                latex_tbl = latex_tables[t_idx]
                try:
                    record["latex_verification"] = _verify_structured_with_latex(
                        ctx,
                        parsed_root=parsed_root,
                        vlm=vlm_client,
                        structured_json=record["structured"],  # type: ignore[arg-type]
                        latex_table=latex_tbl,
                    )
                except Exception as exc:  # noqa: BLE001
                    record["latex_verification"] = {"error": str(exc)}
        tables.append(record)
    return {"paper_id": arxiv_id.name, "tables": tables}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为单篇论文构建基于表格的多跳 hop 结构（依赖 VLM）")
    parser.add_argument("--paper", type=Path, required=True, help="mineru 处理后的论文目录（arxiv_id 或其下的 vlm）")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o-mini", help="VLM 模型名（serve 接口使用）")
    parser.add_argument("--vlm-base-url", type=str, default="http://localhost:8000/v1", help="已有 vLLM serve 的 base_url（如 http://localhost:8000/v1）")
    parser.add_argument("--vlm-api-key", type=str, default=None, help="serve 的 api_key")
    parser.add_argument("--vlm-launch-server", action="store_true", help="是否在本进程内启动 vllm serve")
    parser.add_argument("--vlm-model-path", type=str, default=None, help="启动 vllm serve 时的模型路径")
    parser.add_argument("--vlm-gpus", type=str, default=None, help="启动 vllm serve 使用的 GPU，逗号分隔，默认前四张 0,1,2,3")
    parser.add_argument("--vlm-port", type=int, default=8000, help="启动 vllm serve 的端口")
    parser.add_argument("--window", type=int, default=10, help="表格前后收集的 text 数量")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hops = build_paper_hops(
        paper_dir=args.paper,
        vlm_model=args.vlm_model,
        vlm_base_url=args.vlm_base_url,
        vlm_api_key=args.vlm_api_key,
        vlm_launch_server=args.vlm_launch_server,
        vlm_model_path=args.vlm_model_path,
        vlm_gpus=args.vlm_gpus,
        vlm_port=args.vlm_port,
        window=args.window,
        body_only=args.body_only,
        verify_with_latex=args.verify_with_latex,
        latex_base=args.latex_base,
    )
    print(json.dumps(hops, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
