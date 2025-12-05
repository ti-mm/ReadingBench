import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# ================= 配置区域 =================
INPUT_DIR = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/multihop_qa/Table_Json")
OUTPUT_CSV = Path("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/tests/judge_json.csv")
# ===========================================

def load_json(file_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return {}

def determine_status(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    返回 (should_record, failure_type)
    """
    verify = data.get("latex_verification", {})
    
    # 情况0: 验证过程直接报错
    if "error" in verify:
        return True, "ERROR_IN_VERIFICATION"

    same_table = verify.get("same_table")
    content_match = verify.get("content_match")

    is_same_false = (same_table is False)
    is_match_false = (content_match is False)

    # 情况1: 两者都为 False
    if is_same_false and is_match_false:
        return True, "BOTH_FALSE"
    
    # 情况2: 只有 same_table 为 False
    if is_same_false:
        return True, "SAME_TABLE_FALSE_ONLY"
    
    # 情况3: 只有 content_match 为 False
    if is_match_false:
        return True, "CONTENT_MATCH_FALSE_ONLY"
    
    # 通过
    return False, "PASS"

def main():
    if not INPUT_DIR.exists():
        print(f"[Error] Input directory does not exist: {INPUT_DIR}")
        return

    headers = [
        "failure_type",
        "arxiv_id",
        "file_name", 
        "page_idx", 
        "table_entry_index", 
        "latex_match_score",
        "same_table", 
        "content_match", 
        "problems"
    ]
    
    records = []
    # 用于统计所有状态的计数器
    all_stats = Counter()
    total_valid_files = 0
    
    print(f"Scanning directory: {INPUT_DIR} ...")
    
    json_files = sorted(INPUT_DIR.rglob("*.json"))
    
    for file_path in json_files:
        data = load_json(file_path)
        if not data:
            continue
        
        # 有效文件计数 +1
        total_valid_files += 1
            
        should_record, fail_type = determine_status(data)
        
        # 记录状态（无论是 PASS 还是各种错误）
        all_stats[fail_type] += 1
        
        if should_record:
            verify = data.get("latex_verification", {})
            problems = verify.get("problems", [])
            problems_str = "; ".join(str(p) for p in problems) if isinstance(problems, list) else str(problems)
            
            record = {
                "failure_type": fail_type,
                "arxiv_id": file_path.parent.name,
                "file_name": file_path.name,
                "page_idx": data.get("page_idx"),
                "table_entry_index": data.get("table_entry_index"),
                "latex_match_score": data.get("latex_match_score"),
                "same_table": verify.get("same_table"),
                "content_match": verify.get("content_match"),
                "problems": problems_str
            }
            records.append(record)

    # 写入 CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)
        print(f"Results saved to: {OUTPUT_CSV}")
    except Exception as e:
        print(f"[Error] Failed to write CSV: {e}")

    # ================= 打印统计报告 =================
    if total_valid_files > 0:
        print("\n" + "="*40)
        print("          STATISTICS REPORT")
        print("="*40)
        print(f"Total JSON Files: {total_valid_files}")
        
        # 计算成功率
        pass_count = all_stats.get("PASS", 0)
        pass_rate = (pass_count / total_valid_files) * 100
        print(f"Success (PASS):   {pass_count} ({pass_rate:.2f}%)")
        print("-" * 40)
        print("Failure Breakdown:")
        
        # 定义需要展示的错误类型顺序
        fail_types = [
            "BOTH_FALSE", 
            "SAME_TABLE_FALSE_ONLY", 
            "CONTENT_MATCH_FALSE_ONLY", 
            "ERROR_IN_VERIFICATION"
        ]
        
        total_fail_count = 0
        for f_type in fail_types:
            count = all_stats.get(f_type, 0)
            if count > 0:
                rate = (count / total_valid_files) * 100
                print(f"  - {f_type:<24}: {count} ({rate:.2f}%)")
                total_fail_count += count
        
        print("-" * 40)
        total_fail_rate = (total_fail_count / total_valid_files) * 100
        print(f"Total Failures:   {total_fail_count} ({total_fail_rate:.2f}%)")
        print("="*40 + "\n")
    else:
        print("\n[Warning] No valid JSON files processed.")

if __name__ == "__main__":
    main()