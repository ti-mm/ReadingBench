import sys
import os
from pathlib import Path

# --- 路径配置 ---
PROJECT_ROOT = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench")
INPUT_DIR = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/parsed_pdfs")
OUTPUT_DIR = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/multihop_qa/Table_Json_VLLM")

# 将项目根目录加入 python path，确保能导入 QA 模块
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # 导入刚才修改好的 mapping_vlm 中的核心函数
    # 注意：这依赖于你已经保存了上一轮回复中的完整 mapping_vlm.py
    from QA.multihop_qa.mapping_vlm import build_paper_hops
except ImportError as e:
    print("错误：无法导入 mapping_vlm，请检查文件位置或 sys.path 设置。")
    print(f"详细信息: {e}")
    sys.exit(1)

def main():
    # --- vLLM 参数配置 ---
    # 这些参数将传递给 build_paper_hops
    vllm_config = {
        "extract_vlm_base_url": "http://localhost:8000/v1",
        "extract_vlm_model": "qwen-vl",  # 对应 vllm serve --served-model-name
        "verify_with_latex": True,       # 是否开启 Latex 验证
        # 如果需要指定 latex 源码目录，可取消下面注释并修改路径
        # "latex_base": Path("...") 
    }

    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        print(f"错误: 输入目录不存在 {INPUT_DIR}")
        return

    # 1. 获取所有子目录并按名称排序
    all_items = sorted(INPUT_DIR.iterdir())
    paper_dirs = [p for p in all_items if p.is_dir()]
    
    # 2. 取前 10 个
    target_papers = paper_dirs[:10]
    
    print(f"扫描到 {len(paper_dirs)} 个目录，将处理前 {len(target_papers)} 个。")
    
    # 确保输出根目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 循环处理
    for i, paper_path in enumerate(target_papers):
        paper_id = paper_path.name
        print(f"\n[{i+1}/{len(target_papers)}] 正在处理: {paper_id} ...")
        
        try:
            # 调用 mapping_vlm.py 中的逻辑
            # build_paper_hops 内部会自动处理文件保存 (只要使用了之前提供的修改版代码)
            result = build_paper_hops(
                paper_dir=paper_path,
                save_dir=OUTPUT_DIR,
                **vllm_config
            )
            
            # 简单的结果统计
            table_count = len(result.get("tables", []))
            print(f"   -> 完成。提取到 {table_count} 个表格。")
            
        except Exception as e:
            print(f"   -> 处理失败: {e}")
            # 打印简短堆栈以便调试
            import traceback
            traceback.print_exc()

    print("\n==================================")
    print("所有任务执行完毕。")

if __name__ == "__main__":
    main()