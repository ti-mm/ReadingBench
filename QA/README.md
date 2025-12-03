# QA Workflow Overview

This repo slice uses MinerU outputs to build table-centric structured data for multi-hop QA experiments. The current flow keeps only the “structured table” path and removes VLM-based question generation.

## Inputs
- MinerU parsed PDF folder: `parsed_pdfs/{arxiv_id}/vlm/{arxiv_id}_content_list.json` plus table images.
- (Optional) LaTeX source for the same paper: `test_database/latex_src/{arxiv_id}` or a custom `--latex-base`.

## Main script
`python QA/multihop_qa/mapping_vlm.py --paper QA/test_database/parsed_pdfs/1610.02136`

Output: JSON with one entry per table containing:
- `structured`: VLM-rendered structured JSON of the table image (or `structured_error` on failure).
- `latex_verification`: present only if `--verify-with-latex` is used, reporting `same_table`, `content_match`, `problems`, `summary`.

## Options
- `--verify-with-latex`  
  Compare the VLM-structured JSON against the LaTeX-extracted table (uses `mapping_llm.extract_tables_from_package` trimmed to body tables).
- `--latex-base PATH`  
  Override the root for LaTeX sources (default tries `test_database/latex_src/{arxiv_id}`).
- `--all-tables`  
  Disable body-only cutoff; process every table in `content_list`. Default behavior stops at the first `ref_text` or a `text` item equal to `references`/`appendix`.
- Body-only is on by default; use `--all-tables` to include appendix/reference tables.

## VLM connection
- `--vlm-model`, `--vlm-base-url`, `--vlm-api-key`
- `--vlm-launch-server` with `--vlm-model-path`, `--vlm-gpus`, `--vlm-port`

## Workflow summary
1) Load MinerU `content_list` and keep正文 tables (unless `--all-tables`).  
2) For each table image, call VLM with the hierarchical-column prompt to get structured JSON.  
3) (Optional) Load the aligned LaTeX table and ask VLM to judge table identity + content match, recording issues.  
4) Emit a JSON object: `{"paper_id": "...", "tables": [...]}`.***
