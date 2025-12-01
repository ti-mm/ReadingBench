import os
import json
from pathlib import Path
from loguru import logger

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode


def parse_many_pdfs(pdf_paths: list[str], output_root: str,
                    draw_layout=True, draw_span=False, model_path=None):
    """
    pdf_paths : list of paths to PDFs
    output_root: root directory to put outputs. For each PDF, create subfolder.
    """
    for pdf_path in pdf_paths:
        try:
            logger.info(f"=== Parsing {pdf_path} ===")
            file_name = Path(pdf_path).stem
            pdf_bytes = read_fn(pdf_path)
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

            local_image_dir, local_md_dir = prepare_env(output_root, file_name, "vlm")
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)

            extra_args = {}
            if model_path:
                extra_args["model_path"] = model_path
                logger.info(f"Using VLM model: {model_path}")

            logger.info("Running vlm_doc_analyze ...")
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend="vllm-engine",
                **extra_args
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


if __name__ == "__main__":
    os.environ['MINERU_MODEL_SOURCE'] = "local"

    base_dir = "/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/pdfs"
    pdf_list = [os.path.join(base_dir, p) for p in os.listdir("/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/pdfs")]
    parse_many_pdfs(
        pdf_paths=pdf_list,
        output_root="/inspire/hdd/project/embodied-multimodality/lujiahao-253108120106/workspace/ReadingBench/QA/test_database/parsed_pdfs",
        draw_layout=True,
        draw_span=False,
        model_path="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Qwen3-VL-30B-A3B-Instruct"  # or "/path/to/your/vlm/model"
    )