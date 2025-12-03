import os
import sys
from pathlib import Path

from pylatexenc.latexwalker import LatexEnvironmentNode, LatexWalker

# 将 QA 目录加入 sys.path，便于导入 multihop_qa 包
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multihop_qa.latex_tables_pylatexenc import merge_latex_dir

# 定义可能的表格环境名称
TABLE_ENVS = ("table", "tabular", "longtable", "tabularx", "table*", "tabu")


def find_all_table_envs(nodelist):
    """递归找到所有表格环境节点，不管嵌套"""
    envs = []
    for node in nodelist:
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname.lower() in TABLE_ENVS:
                envs.append(node)
            envs.extend(find_all_table_envs(node.nodelist))
        else:
            if hasattr(node, "nodelist") and node.nodelist:
                envs.extend(find_all_table_envs(node.nodelist))
    return envs


def filter_top_level(envs):
    """过滤掉被其他环境完全包含的嵌套环境，只保留最外层"""
    top_level = []
    for env in envs:
        e_start = env.pos
        e_end = env.pos + env.len
        is_nested = False
        for other in envs:
            if env is other:
                continue
            o_start = other.pos
            o_end = other.pos + other.len
            if o_start <= e_start and e_end <= o_end:
                is_nested = True
                break
        if not is_nested:
            top_level.append(env)
    return top_level


def scan_real_package(tex_dir: Path):
    merged, main_tex = merge_latex_dir(tex_dir)
    walker = LatexWalker(merged)
    nodelist, _, _ = walker.get_latex_nodes()
    all_envs = find_all_table_envs(nodelist)
    top_envs = filter_top_level(all_envs)
    print(f"[{tex_dir.name}] main_tex={main_tex.name}, total_envs={len(all_envs)}, top_tables={len(top_envs)}")
    for idx, env in enumerate(top_envs, 1):
        snippet = env.latex_verbatim()
        # snippet = snippet[:400] + (" ...\n" if len(snippet) > 400 else "\n")
        print(f"  Table {idx}:")
        print(snippet)
    print("-" * 60)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    latex_root = repo_root / "test_database" / "latex_src"
    if not latex_root.exists():
        print(f"latex_root not found: {latex_root}")
        return

    arxiv_env = os.environ.get("ARXIV_ID")
    arxiv_env = "1804.00819"
    targets = []
    if arxiv_env:
        candidate = latex_root / arxiv_env
        if candidate.exists():
            targets = [candidate]
    if not targets:
        targets = [p for p in sorted(latex_root.iterdir()) if p.is_dir()][:3]

    for tex_dir in targets:
        try:
            scan_real_package(tex_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[{tex_dir.name}] failed: {exc}")


if __name__ == "__main__":
    main()
