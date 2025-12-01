import os
import json
from pathlib import Path
from loguru import logger

# 引入 vllm 用于手动加载模型
try:
    from vllm import LLM, SamplingParams
except ImportError:
    logger.error("Please install vllm: pip install vllm")
    exit(1)

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode


def parse_many_pdfs(pdf_paths: list[str], output_root: str, vllm_engine,
                    draw_layout=True, draw_span=False):
    """
    pdf_paths : list of paths to PDFs
    output_root: root directory to put outputs.
    vllm_engine: Initialized vLLM object
    """
    for pdf_path in pdf_paths:
        try:
            logger.info(f"=== Parsing {pdf_path} ===")
            file_name = Path(pdf_path).stem
            pdf_bytes = read_fn(pdf_path)
            # 转换 PDF
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

            local_image_dir, local_md_dir = prepare_env(output_root, file_name, "vlm")
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)

            logger.info("Running vlm_doc_analyze ...")
            
            # 关键修改：直接传递 vllm_llm 对象，而不是传递 model_path
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend="vllm-engine",
                vllm_llm=vllm_engine  # 传入已加载的模型实例
            )
            logger.info("VLM inference done")

            pdf_info = middle_json["pdf_info"]

            # 保存原始 PDF
            md_writer.write(f"{file_name}_origin.pdf", pdf_bytes)

            if draw_layout:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{file_name}_layout.pdf")
            if draw_span:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{file_name}_span.pdf")

            md_writer.write_string(
                f"{file_name}_middle.json",
                json.dumps(middle_json, ensure_ascii=False, indent=4)
            )
            md_writer.write_string(
                f"{file_name}_model.json",
                json.dumps(infer_result, ensure_ascii=False, indent=4)
            )

            content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, Path(local_image_dir).name)
            md_writer.write_string(
                f"{file_name}_content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=4)
            )

            md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, Path(local_image_dir).name)
            md_writer.write_string(f"{file_name}.md", md_content)

            logger.success(f"Finished {pdf_path}, output in {local_md_dir}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈以便调试


if __name__ == "__main__":
    os.environ['MINERU_MODEL_SOURCE'] = "local"
    
    # 路径配置
    base_dir = "/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/pdfs"
    output_dir = "/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/parsed_pdfs"
    model_path = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen2-VL-72B-Instruct"

    # 获取 PDF 列表
    if not os.path.exists(base_dir):
        logger.error(f"Base dir not found: {base_dir}")
        exit(1)
        
    pdf_list = [os.path.join(base_dir, p) for p in os.listdir(base_dir) if p.endswith('.pdf')]
    
    logger.info(f"Found {len(pdf_list)} PDFs. Loading Model (This runs only once)...")

    # 1. 在主进程初始化模型 (解决重复加载问题)
    # tensor_parallel_size 根据你的 GPU 数量调整，如果只有一张卡设为1，多张卡(如4张)设为4
    import torch
    gpu_count = torch.cuda.device_count()
    
    llm = LLM(
        model=model_path, 
        trust_remote_code=True, 
        tensor_parallel_size=gpu_count, # 自动使用所有可见 GPU
        gpu_memory_utilization=0.9,
        max_model_len=8192 # 根据显存大小适当调整，太大会OOM
    )
    
    logger.success("Model loaded successfully.")

    # 2. 开始批处理
    parse_many_pdfs(
        pdf_paths=pdf_list,
        output_root=output_dir,
        vllm_engine=llm, # 传递模型实例
        draw_layout=True,
        draw_span=False
    )