import os
import json
import time
import random
import html
import re
import requests
from pathlib import Path
from typing import Any, Dict, Optional
from bs4 import BeautifulSoup
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


# -----------------------------------
# Main pipeline
# -----------------------------------

def main():
    base_dir = "/Users/tim/Desktop/arxiv_spider/QA/test_database"
    ids_file = f"{base_dir}/candidate_arxiv_ids.json"
    metadata_out = f"{base_dir}.metadata.jsonl"
    pdf_dir = f"{base_dir}/pdfs"

    os.makedirs(pdf_dir, exist_ok=True)

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

            # write metadata jsonl
            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # 2. download pdf
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_path = f"{pdf_dir}/{arxiv_id}.pdf"

            if not os.path.exists(pdf_path):
                try:
                    pdf_resp = _polite_get(pdf_url)
                    with open(pdf_path, "wb") as pdf_f:
                        pdf_f.write(pdf_resp.content)
                    logger.info(f"PDF saved: {pdf_path}")
                except Exception as exc:
                    logger.error(f"Failed to download pdf for {arxiv_id}: {exc}")

            time.sleep(random.uniform(0.3, 0.8))

    logger.info("All done.")


if __name__ == "__main__":
    main()