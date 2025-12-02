import os
import json
import time
import random
import html
import re
import tarfile
import shutil
import requests
from pathlib import Path
from typing import Any, Dict, Optional
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse

# ---------------------------
# Your helper functions
# ---------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; remote-paper-fetcher/1.0; +agentschat)",
        }
    )
    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        # allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()
_CLEAN_ABS_PREFIX_RE = re.compile(r"^\s*(Abstract|ABSTRACT)\s*[:\-]*\s*", flags=re.IGNORECASE)


def _polite_get(url: str, timeout: float = 5.0, min_sleep: float = 0.4, max_sleep: float = 1.2) -> requests.Response:
    response = _SESSION.get(url, timeout=timeout)
    print(response)
    if response.status_code == 429:
        time.sleep(2.0 + random.random())

    time.sleep(random.uniform(min_sleep, max_sleep))
    response.raise_for_status()
    return response


def _clean_abstract(text: str) -> str:
    if not text:
        return ""
    normalized = html.unescape(text).replace("\xa0", " ")
    normalized = _CLEAN_ABS_PREFIX_RE.sub("", normalized).strip()
    normalized = re.split(r"\s+Authors?:|Subjects?:", normalized, maxsplit=1)[0].strip()
    return normalized


def _extract_abstract(soup: BeautifulSoup) -> Optional[str]:
    meta_abs = soup.select_one("meta[name='citation_abstract']")
    if meta_abs and meta_abs.get("content", "").strip():
        return _clean_abstract(meta_abs["content"].strip())

    abs_block = soup.find("blockquote", class_="abstract") or soup.find("div", class_="abstract")
    if abs_block:
        return _clean_abstract(abs_block.get_text(" ", strip=True))

    meta_desc = soup.select_one("meta[name='description']") or soup.select_one("meta[property='og:description']")
    if meta_desc and meta_desc.get("content", "").strip():
        return _clean_abstract(meta_desc["content"].strip())

    return None


def _scrape_abs_page(arxiv_id: str, timeout: float = 15.0) -> Optional[Dict[str, Any]]:
    url = f"https://arxiv.org/abs/{arxiv_id}"

    try:
        response = _polite_get(url, timeout=timeout)
    except Exception as exc:
        logger.warning(f"Failed to fetch arXiv abs page {arxiv_id}: {exc}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # title
    title = None
    title_meta = soup.select_one("meta[name='citation_title']")
    if title_meta and title_meta.get("content"):
        title = title_meta["content"].strip()
    else:
        heading = soup.find("h1", class_="title")
        if heading:
            title = heading.get_text(strip=True).replace("Title:", "").strip()

    # authors
    authors = [node["content"].strip() for node in soup.select("meta[name='citation_author']") if node.get("content")]
    if not authors:
        authors_block = soup.find("div", class_="authors")
        if authors_block:
            text = authors_block.get_text(" ", strip=True).replace("Authors:", "").strip()
            authors = [p.strip() for p in text.split(",") if p.strip()]

    summary = _extract_abstract(soup)

    return {
        "id": arxiv_id,
        "title": title or "",
        "authors": authors,
        "summary": summary or "",
    }


def _download_and_extract_latex(arxiv_id: str, latex_dir: Path) -> Optional[Path]:
    """
    下载 arxiv 源码 tar.gz 并解压，返回解压后的子目录路径；若失败返回 None。
    """
    url = f"https://arxiv.org/src/{arxiv_id}"
    target_dir = latex_dir / arxiv_id
    tmp_dir = latex_dir / f"{arxiv_id}__tmp"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tar_path = tmp_dir / f"{arxiv_id}.tar.gz"
    try:
        resp = _polite_get(url, timeout=30.0)
        tar_path.write_bytes(resp.content)
    except Exception as exc:
        logger.warning(f"Failed to download latex src for {arxiv_id}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)
        tar_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Failed to extract latex src for {arxiv_id}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # 将 tmp_dir 提升为正式目录，确保失败时不会留下空目录
    tmp_dir.rename(target_dir)

    # 只保留子文件夹，若存在直接子文件则保留在 target_dir。
    # 返回顶级子目录路径（若存在唯一子目录），否则返回 target_dir。
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return target_dir


# -----------------------------------
# Main pipeline
# -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv metadata, PDFs, and LaTeX sources.")
    parser.add_argument("--base-dir", type=Path, default=Path("/Users/tim/Downloads/ReadingBench/QA/test_database"))
    parser.add_argument("--ids-file", type=Path, default=Path("/Users/tim/Downloads/ReadingBench/QA/test_database/candidate_arxiv_ids.json"), help="Path to candidate_arxiv_ids.json")
    args = parser.parse_args()

    base_dir = args.base_dir
    ids_file = args.ids_file or (base_dir / "candidate_arxiv_ids.json")
    metadata_out = base_dir / "metadata.jsonl"
    pdf_dir = base_dir / "pdfs"
    latex_dir = base_dir / "latex_src"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # load arxiv id list
    with open(ids_file, "r") as f:
        arxiv_ids = json.load(f)
    logger.info(f"Loaded {len(arxiv_ids)} arxiv ids")

    # open jsonl writer
    with open(metadata_out, "a", encoding="utf-8") as fout:

        for arxiv_id in arxiv_ids:
            logger.info(f"Processing {arxiv_id}")

            # 1. scrape metadata
            meta = _scrape_abs_page(arxiv_id)
            if meta is None:
                logger.error(f"Metadata failed for {arxiv_id}")
                continue

            # 2. download pdf
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_path = pdf_dir / f"{arxiv_id}.pdf"

            if not pdf_path.exists():
                try:
                    pdf_resp = _polite_get(pdf_url)
                    pdf_path.write_bytes(pdf_resp.content)
                    logger.info(f"PDF saved: {pdf_path}")
                except Exception as exc:
                    logger.error(f"Failed to download pdf for {arxiv_id}: {exc}")

            # 3. download and extract LaTeX source（若不存在再下载）
            latex_candidate = latex_dir / arxiv_id
            if latex_candidate.exists() and not any(latex_candidate.iterdir()):
                # 清理掉空目录，防止误判为已下载
                shutil.rmtree(latex_candidate, ignore_errors=True)

            latex_path: Optional[Path] = latex_candidate if latex_candidate.exists() else None
            if latex_path is None:
                try:
                    latex_path = _download_and_extract_latex(arxiv_id, latex_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"LaTeX src failed for {arxiv_id}: {exc}")
                    latex_path = None

            meta["latex_path"] = str(latex_path) if latex_path and Path(latex_path).exists() else None

            # add pdf/parsed paths
            meta["pdf_path"] = str(pdf_path) if pdf_path.exists() else None
            parsed_dir = base_dir / "parsed_pdfs" / arxiv_id
            meta["parsed_path"] = str(parsed_dir) if parsed_dir.exists() else None

            # write metadata jsonl
            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

            time.sleep(random.uniform(0.3, 0.8))

    logger.info("All done.")


if __name__ == "__main__":
    main()
