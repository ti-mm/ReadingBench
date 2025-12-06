#!/bin/bash

# ================= 绝对路径配置区域 =================

# 1. 项目根路径 (Workspace Root)
PROJECT_ROOT="/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench"

# 2. 脚本与数据路径
SCRIPT_PATH="${PROJECT_ROOT}/QA/multihop_qa/mapping_vlm.py"
PARSED_PDFS_DIR="${PROJECT_ROOT}/QA/test_database/parsed_pdfs"

# 【重要修改】Latex 基准目录
# Python 脚本内部会自动寻找 "base_dir/test_database/latex_src/{id}"
# 所以这里我们指向 QA 根目录即可，避免路径重复拼接导致找不到文件
LATEX_SRC_DIR="${PROJECT_ROOT}/QA"

# 3. 输出结果保存路径
OUTPUT_SAVE_ROOT="${PROJECT_ROOT}/QA/multihop_qa/Table_Json"

# 4. vLLM 模型与服务配置
# 【已确认】与服务端启动命令保持一致 (无 -FP8 后缀)
VLM_MODEL_PATH="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-VL-235B-A22B-Instruct-FP8"
API_URL="http://localhost:8000/v1"
API_KEY="EMPTY"

# ================= 环境变量设置 =================
# 将 QA 目录加入 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/QA":$PYTHONPATH

# ================= 执行逻辑 =================
echo "=================================================="
echo "开始批量处理任务"
echo "脚本路径: $SCRIPT_PATH"
echo "输入目录: $PARSED_PDFS_DIR"
echo "Latex基准: $LATEX_SRC_DIR"
echo "输出目录: $OUTPUT_SAVE_ROOT"
echo "=================================================="

# 遍历输入目录下的所有子文件夹
for paper_full_path in "$PARSED_PDFS_DIR"/*; do
    if [ -d "$paper_full_path" ]; then
        # 获取文件夹名称 (即 arxiv_id)
        paper_id=$(basename "$paper_full_path")
        
        # 拼接目标输出文件夹的绝对路径
        target_output_dir="${OUTPUT_SAVE_ROOT}/${paper_id}"
        
        # 检查该 output 目录是否已存在
        if [ -d "$target_output_dir" ]; then
            echo ">> [跳过] ${paper_id} : 目录已存在 -> $target_output_dir"
            continue
        fi

        echo "--------------------------------------------------"
        echo ">> [开始] 正在处理: ${paper_id}"
        echo "   来源: ${paper_full_path}"
        
        # 执行 Python 脚本
        # --extract-vlm-api-key 显式传入 "EMPTY" 以防万一
        python "$SCRIPT_PATH" \
            --paper "$paper_full_path" \
            --extract-vlm-model "$VLM_MODEL_PATH" \
            --extract-vlm-base-url "$API_URL" \
            --extract-vlm-api-key "$API_KEY" \
            --verify-with-latex \
            --latex-base "$LATEX_SRC_DIR" \
            --body-only
        
        # 检查退出代码
        if [ $? -eq 0 ]; then
            echo ">> [成功] ${paper_id}"
        else
            echo ">> [错误] ${paper_id} 处理失败，请检查日志。"
        fi
    fi
done

echo "--------------------------------------------------"
echo "所有任务已完成。"