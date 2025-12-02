from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TableContext:
    """表格的基本上下文信息，供 mapping.py 使用。"""

    arxiv_id: str
    page_idx: int
    table_entry_index: int
    image_path: str
    table_caption: List[str]
    table_footnote: List[str]
    table_body: str
    text_before: List[str]
    text_after: List[str]

    def nearby_text(self) -> str:
        return "\n".join(self.text_before + self.text_after)

    def image_full_path(self, parsed_root: Path) -> Path:
        return parsed_root / self.image_path
