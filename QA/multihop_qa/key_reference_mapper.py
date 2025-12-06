"""
基于 vLLM（本地 Python 或 serve）将 mapping 结果中的 key（数据集/方法名称）映射到参考文献。

流程（每个 key）：
1) 从 ../test_database/parsed_pdfs/{arxiv_id}/vlm/{arxiv_id}_content_list.json 中提取全部 ref_text。
2) 将 refs 按块送入 LLM，请求返回是否有匹配，以及匹配的参考文献/标题（JSON 格式）。
3) 汇总所有块的候选；若有多个候选，再用 LLM 进行一次消歧，选出最佳。
4) 返回的 key_reference_map 形如：
   {
     key: {
       "matched": true/false,
       "selection": "选中的参考文献文本或标题",
       "candidates": ["候选1", "候选2", ...]
     },
     ...
   }

额外：会对 mapping_vlm 中 latex_verification.same_table 与 content_match 均为 true 的表格，
调用 LLM 从结构化表格 JSON 中提取可能的“方法/数据集”名称，输出 verified_table_candidates。

注意：仅支持 OpenAI 兼容服务（如 vLLM serve），通过 llm_client.ServeLLMClient 调用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from multihop_qa.llm_client import BaseLLMClient, ServeLLMClient, ServeLLMConfig

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def extract_ref_texts(arxiv_id: str, base_dir: Path) -> List[str]:
    """
    从 ../test_database/parsed_pdfs/{arxiv_id}/vlm/{arxiv_id}_content_list.json
    中提取所有 type == "ref_text" 的文本列表。
    """
    content_path = base_dir / "test_database" / "parsed_pdfs" / arxiv_id / "vlm" / f"{arxiv_id}_content_list.json"
    if not content_path.exists():
        return []
    try:
        items = json.loads(content_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    def _is_ref_text(entry: Dict) -> bool:
        return entry.get("type") == "ref_text" or entry.get("sub_type") == "ref_text"

    def _get_text(entry: Dict) -> str:
        if entry.get("text"):
            return str(entry.get("text"))
        if isinstance(entry.get("list_items"), list):
            return "\n".join(str(x) for x in entry["list_items"])
        return ""

    refs: List[str] = []
    prev_ref_text: Optional[str] = None
    for item in items:
        if not _is_ref_text(item):
            prev_ref_text = None
            continue

        curr_text = _get_text(item)
        if not curr_text:
            prev_ref_text = None
            continue

        # 构造上下文增强块：如果前一个也是 ref_text，则拼接其末尾 3 行（或末尾 400 字符）。
        merged = curr_text
        if prev_ref_text:
            lines = prev_ref_text.splitlines()
            tail_lines = "\n".join(lines[-3:]) if lines else ""
            tail_chars = prev_ref_text[-400:]
            tail = tail_lines or tail_chars
            merged = tail + ("\n" if tail else "") + curr_text

        refs.append(merged)
        prev_ref_text = curr_text

    return refs


def _chunk_list(seq: Sequence, chunk_size: int) -> Iterable[List]:
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i : i + chunk_size])


def _match_chunk(llm: BaseLLMClient, key: str, refs_chunk: List[str]) -> Dict:
    prompt = (
        "你是论文引用匹配助手。给定一个 key（可能是方法名或数据集名）和若干参考文献条目，"
        "判断是否有条目与该 key 对应。返回纯 JSON，格式："
        '{"found": true/false, "candidates": ["引用文本或标题", ...]}. '
        "只返回 JSON，不要额外文本。\n\n"
        f"key: {key}\n"
        f"refs:\n- " + "\n- ".join(refs_chunk)
    )
    return llm.ask_json(prompt)


def _disambiguate(llm: BaseLLMClient, key: str, candidates: List[str]) -> Dict:
    prompt = (
        "给定 key 和多个可能的参考文献候选，选出最匹配的一个。返回纯 JSON："
        '{"selection": "候选文本", "reason": "..."}。只返回 JSON。\n\n'
        f"key: {key}\n候选：\n- " + "\n- ".join(candidates)
    )
    return llm.ask_json(prompt)


def _verification_ok(record: Dict) -> bool:
    """
    仅依据 latex_verification.same_table 与 content_match 判定是否处理。
    """
    if not isinstance(record, dict):
        return False
    verify = record.get("latex_verification", {})
    return bool(verify and verify.get("same_table") is True and verify.get("content_match") is True)


def _verified_table_prompt(table_idx: int, structured_table: Dict) -> str:
    table_json = json.dumps(structured_table, ensure_ascii=False)
    return (
        "你会看到一张已经通过 LaTeX 校验（same_table=true 且 content_match=true）的表格 JSON。"
        "请找出其中可能代表“方法/模型/算法名称”或“数据集/任务名称”的词语，过滤掉度量指标/列名（如 Accuracy、F1、Params、Epoch）。"
        "只返回纯 JSON，不要解释，格式："
        '{"candidates": [{"name": "...", "type": "method|dataset|other", "reason": "..."}]}。\n\n'
        f"表格索引: {table_idx}\n表格 JSON:\n{table_json}"
    )


def _extract_candidates_from_verified_tables(
    mapping_result: Dict, llm_client: BaseLLMClient, max_tokens: int = 800
) -> Dict[str, Dict]:
    """
    针对 mapping_vlm 输出中 latex_verification.same_table/content_match 均为 True 的表格，
    调用 LLM 识别其中可能的“方法/数据集”名称。
    返回一个 dict，key 为表格顺序索引（字符串），value 包含候选列表等信息。
    """
    table_candidates: Dict[str, Dict] = {}
    tables = mapping_result.get("tables", [])
    for idx, table in enumerate(tables):
        verification: Optional[Dict] = table.get("latex_verification")
        structured = table.get("structured")
        if not verification or verification.get("same_table") is not True or verification.get("content_match") is not True:
            continue
        if not isinstance(structured, dict):
            continue

        try:
            resp = llm_client.ask_json(_verified_table_prompt(idx, structured), max_tokens=max_tokens)
            raw_candidates = resp.get("candidates", [])
        except Exception as exc:  # noqa: BLE001
            table_candidates[str(idx)] = {
                "table_order_index": table.get("table_order_index", idx),
                "table_entry_index": table.get("table_entry_index"),
                "caption": structured.get("caption"),
                "candidates": [],
                "error": str(exc),
            }
            continue

        parsed: List[Dict[str, str]] = []
        if isinstance(raw_candidates, list):
            seen = set()
            for cand in raw_candidates:
                if isinstance(cand, dict):
                    name = str(cand.get("name", "")).strip()
                    if not name:
                        continue
                    norm_name = name.lower()
                    if norm_name in seen:
                        continue
                    seen.add(norm_name)
                    parsed.append(
                        {
                            "name": name,
                            "type": str(cand.get("type", "unknown")),
                            "reason": str(cand.get("reason", "")),
                        }
                    )
                elif isinstance(cand, str):
                    name = cand.strip()
                    if not name:
                        continue
                    norm_name = name.lower()
                    if norm_name in seen:
                        continue
                    seen.add(norm_name)
                    parsed.append({"name": name, "type": "unknown", "reason": ""})

        table_candidates[str(idx)] = {
            "table_order_index": table.get("table_order_index", idx),
            "table_entry_index": table.get("table_entry_index"),
            "caption": structured.get("caption"),
            "candidates": parsed,
        }

    return table_candidates


def map_keys_to_references(
    mapping_result: Dict,
    llm_client: BaseLLMClient,
    base_dir: Path,
    chunk_size: int = 5,
    extract_verified_tables: bool = True,
    verified_table_max_tokens: int = 800,
) -> Dict:
    """
    为 mapping_result 增加 key_reference_map。
    """
    arxiv_id = mapping_result.get("paper_id", "")
    refs = extract_ref_texts(arxiv_id, base_dir)
    keys = set()
    for table in mapping_result.get("tables", []):
        keys.update(table.get("intermediate_hops", {}).keys())

    key_reference_map: Dict[str, Dict] = {}
    for key in keys:
        candidates: List[str] = []
        for chunk in _chunk_list(refs, chunk_size):
            try:
                resp = _match_chunk(llm_client, key, chunk)
            except Exception as exc:  # noqa: BLE001
                # 失败则跳过该 chunk
                continue
            if resp.get("found") and isinstance(resp.get("candidates"), list):
                candidates.extend([str(c) for c in resp["candidates"]])

        if not candidates:
            key_reference_map[key] = {"has_reference": False, "selection": None, "candidates": []}
            continue

        # 去重
        dedup = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                dedup.append(c)

        if len(dedup) == 1:
            key_reference_map[key] = {"has_reference": True, "selection": dedup[0], "candidates": dedup}
        else:
            try:
                disamb = _disambiguate(llm_client, key, dedup)
                selection = disamb.get("selection") or dedup[0]
            except Exception:
                selection = dedup[0]
            key_reference_map[key] = {"has_reference": True, "selection": selection, "candidates": dedup}

    enriched = dict(mapping_result)
    enriched["key_reference_map"] = key_reference_map
    if extract_verified_tables:
        enriched["verified_table_candidates"] = _extract_candidates_from_verified_tables(
            mapping_result, llm_client, max_tokens=verified_table_max_tokens
        )
    return enriched


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="遍历子目录 JSON，按 latex_verification 状态过滤并运行 key-reference 映射")
    parser.add_argument("--input-dir", type=Path, required=True, help="子目录路径，例如 Table_Json/1610.02136")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PACKAGE_ROOT,
        help="数据根目录，用于查找 test_database/parsed_pdfs，默认 QA 根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认覆盖写回 input-dir",
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="仅处理 latex_verification.same_table/content_match 均为 true 的文件",
    )
    parser.add_argument("--chunk-size", type=int, default=5, help="refs 分块大小")
    parser.add_argument(
        "--verified-table-max-tokens",
        type=int,
        default=800,
        help="verified_table_candidates 调用的最大 token 数",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM 模型名（serve 的 served-model-name）")
    parser.add_argument("--llm-base-url", type=str, default="http://localhost:8000/v1", help="LLM 服务 base_url")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM 温度")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"[Error] input-dir is not a directory: {input_dir}")
        return

    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_client = ServeLLMClient(
        ServeLLMConfig(
            model=args.llm_model,
            base_url=str(args.llm_base_url),
            api_key=args.llm_api_key,
            temperature=args.temperature,
        )
    )

    total = 0
    processed = 0
    skipped_unmatched = 0
    errors = 0

    for jf in sorted(input_dir.glob("*.json")):
        total += 1
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[Skip] {jf.name}: read error {exc}")
            continue

        if args.require_match and not _verification_ok(data):
            skipped_unmatched += 1
            print(f"[Skip] {jf.name}: latex_verification not matched (same_table/content_match not both true)")
            continue

        paper_id = data.get("paper_id") or input_dir.name
        mapping_result: Dict[str, object] = {"paper_id": paper_id, "tables": [data]}
        try:
            enriched = map_keys_to_references(
                mapping_result,
                llm_client=llm_client,
                base_dir=args.base_dir,
                chunk_size=args.chunk_size,
                extract_verified_tables=True,
                verified_table_max_tokens=args.verified_table_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[Error] {jf.name}: mapping failed {exc}")
            continue

        out_record = dict(data)
        out_record["paper_id"] = enriched.get("paper_id", paper_id)
        if "key_reference_map" in enriched:
            out_record["key_reference_map"] = enriched.get("key_reference_map")
        if "verified_table_candidates" in enriched:
            out_record["verified_table_candidates"] = enriched.get("verified_table_candidates")

        out_path = output_dir / jf.name
        out_path.write_text(json.dumps(out_record, ensure_ascii=False, indent=2), encoding="utf-8")
        processed += 1
        print(f"[Processed] {jf.name} -> {out_path}")

    print(
        f"Done. scanned={total}, processed={processed}, skipped_unmatched={skipped_unmatched}, errors={errors}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
