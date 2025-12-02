#!/usr/bin/env python3
"""
Generate a JSONL mapping of table images to surrounding text.

The script scans `test_database/parsed_pdfs`, finds table entries in each
`{arxiv_id}_content_list.json`, and captures the 10 preceding and following
text blocks. Output is written to `table_contexts/table_text_contexts.jsonl`
under the QA directory.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = REPO_ROOT / "test_database" / "parsed_pdfs"
OUTPUT_DIR = REPO_ROOT / "table_contexts"
OUTPUT_PATH = OUTPUT_DIR / "table_text_contexts.jsonl"
WINDOW = 18


def collect_text_positions(items: Iterable[Dict]) -> Tuple[List[int], List[str]]:
    positions: List[int] = []
    texts: List[str] = []
    for idx, item in enumerate(items):
        if item.get("type") == "text":
            positions.append(idx)
            texts.append(item.get("text", ""))
    return positions, texts


def table_contexts(items: List[Dict], window: int = WINDOW) -> List[Dict]:
    text_positions, text_values = collect_text_positions(items)
    contexts: List[Dict] = []

    for idx, item in enumerate(items):
        if item.get("type") != "table":
            continue

        insertion_point = bisect.bisect_left(text_positions, idx)
        start = max(0, insertion_point - window)
        end = insertion_point + window

        before = text_values[start:insertion_point]
        after = text_values[insertion_point:end]

        contexts.append(
            {
                "table_entry_index": idx,
                "page_idx": item.get("page_idx"),
                "image_path": item.get("img_path"),
                "table_caption": item.get("table_caption", []),
                "table_footnote": item.get("table_footnote", []),
                "table_body": item.get("table_body", ""),
                "text_before": before[-window:],
                "text_after": after[:window],
            }
        )

    return contexts


def process_arxiv_folder(arxiv_dir: Path) -> List[Dict]:
    vlm_dir = arxiv_dir / "vlm"
    content_path = vlm_dir / f"{arxiv_dir.name}_content_list.json"
    if not content_path.exists():
        return []

    items = json.loads(content_path.read_text(encoding="utf-8"))
    contexts = table_contexts(items, window=WINDOW)

    resolved: List[Dict] = []
    for ctx in contexts:
        image_path = ctx.get("image_path")
        relative_image = None
        if image_path:
            relative_image = str((vlm_dir / image_path).relative_to(PARSED_DIR))

        resolved.append(
            {
                "arxiv_id": arxiv_dir.name,
                "page_idx": ctx.get("page_idx"),
                "table_entry_index": ctx.get("table_entry_index"),
                "image_path": relative_image,
                "table_caption": ctx.get("table_caption", []),
                "table_footnote": ctx.get("table_footnote", []),
                "table_body": ctx.get("table_body", ""),
                "text_before": ctx.get("text_before", []),
                "text_after": ctx.get("text_after", []),
            }
        )

    return resolved


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for arxiv_dir in sorted(PARSED_DIR.iterdir()):
        if not arxiv_dir.is_dir():
            continue
        records.extend(process_arxiv_folder(arxiv_dir))

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=True)
            fh.write("\n")

    print(f"Wrote {len(records)} table context rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
