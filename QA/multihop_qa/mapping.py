#!/usr/bin/env python3
"""
构建基于单篇论文的多跳问答节点。

Hop 划分：
- 中间 hop：从每个表格抽取若干 key/value，对应“数据集/方法”->“与论文的关系”，依赖 VLM。
- 最后一跳：从表格构造问答，答案是方法名/数据集/指标值，依赖 VLM。

输出：列表，每个元素对应一张表格，包含中间 hop dict 和最后一跳的 QA dict。

示例：
    python QA/multihop_qa/mapping.py --paper QA/test_database/parsed_pdfs/1610.02136 --use-vlm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from multihop_qa.models import TableContext
from multihop_qa.vlm_client import VLMClient, VLMConfig

# 默认提取数量
DEFAULT_INTERMEDIATE = 5
DEFAULT_FINAL_QA = 3
MODE_CHOICES = ("intermediate", "final")


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


def _contexts_from_items(items: List[Dict], arxiv_id: str, vlm_dir: Path, parsed_root: Path, window: int = 10) -> List[TableContext]:
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


def _build_intermediate_hops(ctx: TableContext, parsed_root: Path, vlm: VLMClient, max_pairs: int) -> Dict[str, str]:
    prompt = (
        "你是阅读科研论文表格的助手。请从表格中抽取不超过 {n} 个 key-value 对："
        "key 必须是表格中出现的“数据集名称”或“方法名称”；"
        "value 用一句中文描述它和本文的关系（如本文方法、对比方法、最佳方法、使用的数据集、次优方法等），"
        "必要时指出相对排名/是否为本文提出。"
        "仅返回 JSON 对象 {{\"pairs\": [{{\"key\": \"...\", \"relation\": \"...\"}}, ...]}}，不要添加其它内容。"
    ).format(n=max_pairs)
    try:
        data = vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=800)
        pairs = data.get("pairs", [])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 中间 hop VLM 失败: {exc}", file=sys.stderr)
        pairs = []
    result: Dict[str, str] = {}
    for p in pairs:
        key = str(p.get("key", "")).strip()
        rel = str(p.get("relation", "")).strip()
        if key and rel and key not in result:
            result[key] = rel
    return result


def _build_final_hops(ctx: TableContext, parsed_root: Path, vlm: VLMClient, max_qa: int) -> Dict[str, str]:
    prompt = (
        "请基于表格内容生成不超过 {n} 个问答对。"
        "答案必须直接来自表格，可以是方法名称、数据集名称或具体指标值；"
        "问题要明确指出指标/数据集/方法，使答案可核查。"
        "仅返回 JSON 对象 {{\"qa\": [{{\"question\": \"...\", \"answer\": \"...\"}}, ...]}}，不要添加其它内容。"
    ).format(n=max_qa)
    try:
        data = vlm.ask_json(ctx.image_full_path(parsed_root), prompt, max_tokens=800)
        qa_list = data.get("qa", [])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 最后一跳 VLM 失败: {exc}", file=sys.stderr)
        qa_list = []
    result: Dict[str, str] = {}
    for qa in qa_list:
        ans = str(qa.get("answer", "")).strip()
        q = str(qa.get("question", "")).strip()
        if ans and q and ans not in result:
            result[ans] = q
    return result


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
    max_intermediate: int = DEFAULT_INTERMEDIATE,
    max_final: int = DEFAULT_FINAL_QA,
    mode: str = "intermediate",
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
           "intermediate_hops": {key: relation},
           "final_hops": {answer: question}
         }, ...
      ]
    }
    """
    vlm_dir, parsed_root, arxiv_id = _locate_vlm_dir(paper_dir)
    items = _load_content_list(vlm_dir, arxiv_id)
    contexts = _contexts_from_items(items, arxiv_id.name, vlm_dir, parsed_root, window=window)

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
    for ctx in contexts:
        intermediate = {}
        final_hops = {}
        if mode == "intermediate":
            intermediate = _build_intermediate_hops(ctx, parsed_root, vlm_client, max_intermediate)
        elif mode == "final":
            final_hops = _build_final_hops(ctx, parsed_root, vlm_client, max_final)
        tables.append(
            {
                "page_idx": ctx.page_idx,
                "table_entry_index": ctx.table_entry_index,
                "image_path": str(ctx.image_full_path(parsed_root)),
                "intermediate_hops": intermediate,
                "final_hops": final_hops,
            }
        )
    return {"paper_id": arxiv_id.name, "tables": tables}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为单篇论文构建基于表格的多跳 hop 结构（依赖 VLM）")
    parser.add_argument("--paper", type=Path, required=True, help="mineru 处理后的论文目录（arxiv_id 或其下的 vlm）")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o-mini", help="VLM 模型名（serve 接口使用）")
    parser.add_argument("--vlm-base-url", type=str, default=None, help="已有 vLLM serve 的 base_url（如 http://localhost:8000/v1）")
    parser.add_argument("--vlm-api-key", type=str, default=None, help="serve 的 api_key")
    parser.add_argument("--vlm-launch-server", action="store_true", help="是否在本进程内启动 vllm serve")
    parser.add_argument("--vlm-model-path", type=str, default=None, help="启动 vllm serve 时的模型路径")
    parser.add_argument("--vlm-gpus", type=str, default=None, help="启动 vllm serve 使用的 GPU，逗号分隔，默认前四张 0,1,2,3")
    parser.add_argument("--vlm-port", type=int, default=8000, help="启动 vllm serve 的端口")
    parser.add_argument("--window", type=int, default=10, help="表格前后收集的 text 数量")
    parser.add_argument("--max-intermediate", type=int, default=DEFAULT_INTERMEDIATE, help="每张表最多抽取的中间 hop 对数")
    parser.add_argument("--max-final", type=int, default=DEFAULT_FINAL_QA, help="每张表最多生成的最终问答对数")
    parser.add_argument(
        "--mode",
        type=str,
        default="intermediate",
        choices=MODE_CHOICES,
        help="生成中间 hop 还是最终 QA（intermediate|final）",
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
        max_intermediate=args.max_intermediate,
        max_final=args.max_final,
        mode=args.mode,
    )
    print(json.dumps(hops, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
