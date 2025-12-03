#!/usr/bin/env python3
"""
提取 QA/test_database/latex_src 下所有子目录中的表格内容，并写入 jsonl。
成功时记录表格的 env/content/prefix/suffix，失败则记录 error。
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# 将 QA 目录加入 sys.path，便于导入 multihop_qa 包
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multihop_qa.mapping_llm import extract_tables_from_package


def main() -> None:
    repo_root = REPO_ROOT
    latex_root = repo_root / "test_database" / "latex_src"
    output_path = Path(__file__).resolve().parent / "latex_tables.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for subdir in sorted(latex_root.iterdir()):
        if not subdir.is_dir():
            continue
        paper_id = subdir.name
        try:
            tables = extract_tables_from_package(subdir, limit_by_content_list=True)
            for idx, tbl in enumerate(tables):
                records.append(
                    {
                        "paper_id": paper_id,
                        "table_index": idx,
                        "env": tbl.get("env", ""),
                        "content": tbl.get("content", ""),
                        "prefix": tbl.get("prefix", ""),
                        "suffix": tbl.get("suffix", ""),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "paper_id": paper_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
