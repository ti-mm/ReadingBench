import os
import subprocess
import sys
from pathlib import Path

# ================= 配置区域 =================
# 数据集路径
DATA_DIR = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/parsed_pdfs")

# API 配置
QWEN_API_KEY = "sk-915d9967a7f14c89b9b1b6a0cdf0c21c"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-vl-plus"  

# 目标脚本
SCRIPT_PATH = "/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/multihop_qa/mapping_vlm.py"
# ===========================================

def main():
    if not DATA_DIR.exists():
        print(f"[Error] 数据目录不存在: {DATA_DIR}")
        return

    # 获取所有论文文件夹并排序，确保顺序固定
    all_papers = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    
    if not all_papers:
        print("[Error] 没有找到论文文件夹")
        return

    # 固定选取前 3 个
    selected_papers = all_papers[:10]
    
    print(f">>> 开始测试，已修改源码为 Regex 模式")
    print(f">>> 固定选取前 {len(selected_papers)} 篇论文: {[p.name for p in selected_papers]}\n")

    for idx, paper in enumerate(selected_papers, 1):
        print("="*60)
        print(f"[{idx}/{len(selected_papers)}] Processing: {paper.name}")
        print("="*60)
        
        cmd = [
            sys.executable, SCRIPT_PATH,
            "--paper", str(paper),
            "--verify-with-latex",  # 开启 Latex 验证
            "--extract-vlm-model", QWEN_MODEL,
            "--extract-vlm-base-url", QWEN_BASE_URL,
            "--extract-vlm-api-key", QWEN_API_KEY
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed with code {e.returncode}")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    main()