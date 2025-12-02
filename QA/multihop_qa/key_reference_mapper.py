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

注意：仅支持 OpenAI 兼容服务（如 vLLM serve），通过 llm_client.ServeLLMClient 调用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from multihop_qa.llm_client import BaseLLMClient, ServeLLMClient, ServeLLMConfig


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
    refs: List[str] = []
    for item in items:
        if item.get("type") == "ref_text":
            refs.append(item.get("text", ""))
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


def map_keys_to_references(
    mapping_result: Dict,
    llm_client: BaseLLMClient,
    base_dir: Path,
    chunk_size: int = 5,
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
    return enriched
