"""
mixed_layout_extractor.py

Author: Jun Xu

Robust PDF extractor that routes pages with large figures or tables
through GPT-4o Vision and merges the result with classical text extraction.

when processing the pdf files, if some pages contain images or tables, 
the parsing result using the common text extraction tools, e.g., pypdf or pymupdf, is not good enough, 
e.g., missing information or incorrect content.  
Thus, we may write a python code to detect if the page contains table or image (not too small). 
if so, we will convert it into an image, then the code calls vlm api, e.g., gpt-4o, to process it.  
we will compare the result via the image processing and the common text extraction result using llm, 
and then merge into a final result for downstream applications.  

we might use the layout parser to detect the table or image,
and then use the pdf2image to convert the page into an image. However, the models may not be available in the current environment.
https://layout-parser.readthedocs.io/en/latest/notes/installation.html 

"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import fitz  # PyMuPDF            ─ docs: https://pymupdf.readthedocs.io/ :contentReference[oaicite:0]{index=0}
from pdf2image import convert_from_bytes  # reference: https://pdf2image.readthedocs.io/ :contentReference[oaicite:1]{index=1}
import camelot  # quick-start: https://camelot-py.readthedocs.io/ :contentReference[oaicite:2]{index=2}

# Optional deep-layout detector
try:
    import layoutparser as lp  # example: https://layout-parser.readthedocs.io/ :contentReference[oaicite:3]{index=3}
    HAS_LP = True
except ModuleNotFoundError:
    HAS_LP = False


HAS_LP = False


import openai  # image schema & JSON mode: community posts :contentReference[oaicite:4]{index=4}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

AREA_RATIO = 0.05          # ignore images < 2 % of page area (logos etc.)

LINE_RULE_THRESH = 10      # heuristic line count to flag a table
VISION_MODEL = "gpt-4o-mini"
TEXT_MODEL = "gpt-4o-mini"

# You may store the key in env-var OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple on-disk cache for Vision calls
CACHE_DIR = Path(".vision_cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def is_large_bitmap(img_block: Dict, page_rect: fitz.Rect, area_ratio: float = AREA_RATIO) -> bool:
    """
    Decide whether an image block occupies at least *area_ratio* of the page.


    img_block – one element of page.get_text("dict")["blocks"] with ["type"] == 1 
    """

    x0, y0, x1, y1 = img_block["bbox"]
    img_area = (x1 - x0) * (y1 - y0)
    page_area = page_rect.width * page_rect.height
    return (img_area / page_area) >= area_ratio


def camelot_has_table(pdf_bytes: bytes, page_no: int) -> bool:
    """Return True if Camelot successfully extracts ≥ 1 table on *page_no*."""
    try:
        tables = camelot.read_pdf(
            io.BytesIO(pdf_bytes),
            pages=str(page_no),
            flavor="lattice",
            suppress_stdout=True,
        )
        return len(tables) > 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# OpenAI wrappers
# -----------------------------------------------------------------------------

def gpt4o_vision(img: "PIL.Image.Image", prompt: str = "Extract all text and tables") -> str:
    """
    Send *img* to GPT-4o Vision and return plain-text response.
    Follows the base-64 inline image schema. 

    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = openai.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [

                    {"type": "image_url", "image_url": f"url:image/png;base64,{b64}"},

                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


SYSTEM_PROMPT_MERGE = """
You merge two extracts that describe the same PDF page:
(1) vis_text – produced by Vision OCR, rich in tables/figures
(2) raw_text – produced by a PDF text extractor, reliable for plain text

Return strict JSON with keys:
  merged_text  : best reading order text
  omissions    : list of fragments present in only one source
  table_json   : list of tables; each table is a list of rows (each row is list of cells)

Rules:
- Prefer vis_text for tables and images.
- Prefer raw_text for headers, footers, running paragraphs when duplicates occur.
- Do NOT add or hallucinate content.
"""


def gpt_merge(vis_text: str, raw_text: str) -> Dict:
    """
    LLM-assisted reconciliation using JSON-mode (response_format). :contentReference[oaicite:7]{index=7}
    """
    resp = openai.chat.completions.create(
        model=TEXT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MERGE},
            {
                "role": "user",
                "content": f"vis_text<<<{vis_text}>>>raw_text<<<{raw_text}>>>",
            },
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


# -----------------------------------------------------------------------------
# Core extractor
# -----------------------------------------------------------------------------
def extract_pdf_safely(pdf_bytes: bytes) -> List[Dict]:
    """
    Iterate through the PDF and return a list of JSON dicts (one per page) with keys
    merged_text, omissions, table_json.

    Fast pages are handled by PyMuPDF alone; complex pages use Vision + merge.
    """
    results: List[Dict] = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = doc.page_count
    logger.info("Opened PDF with %d pages", num_pages)

    # Optional LayoutParser model
    if HAS_LP:
        model = lp.Detectron2LayoutModel("lp://TableBank/faster_rcnn_R_50_FPN_3x/config")

    for i, page in enumerate(doc, start=1):
        logger.info("Processing page %d/%d", i, num_pages)
        page_dict = page.get_text("dict")
        raw_text = page.get_text()

        # ------------------------------------------------------------------
        # Heuristics: big images?
        big_imgs = [
            b for b in page_dict["blocks"]
            if b["type"] == 1 and is_large_bitmap(b, page.rect)
        ]

        # Heuristics: many ruling lines (table grid)? :contentReference[oaicite:8]{index=8}
        many_rules = len(page_dict.get("lines", [])) > LINE_RULE_THRESH

        # Deep layout model (optional)
        has_table_figure = False
        if HAS_LP and not (big_imgs or many_rules):  # avoid duplicate work
            img_for_lp = page.get_pixmap(dpi=150).tobytes("png")
            lp_layout = model.detect(lp.Image(img_for_lp))
            has_table_figure = any(
                b.type in ("Table", "Figure") for b in lp_layout
            )

        # Camelot quick check (cheap) – if it *succeeds* we can stick to raw_text
        camelot_success = camelot_has_table(pdf_bytes, i)

        use_vision = (
            (big_imgs or many_rules or has_table_figure) and not camelot_success
        )

        # ------------------------------------------------------------------
        if use_vision:
            logger.info("→ Page flagged for Vision processing")
            # Caching by hash
            img_bytes = page.get_pixmap(dpi=300).tobytes("png")
            digest = sha256_bytes(img_bytes)
            cache_file = CACHE_DIR / f"{digest}.txt"

            if cache_file.exists():
                logger.info("  Vision result cached")
                vis_text = cache_file.read_text(encoding="utf-8")
            else:
                vis_text = gpt4o_vision(convert_from_bytes(pdf_bytes,
                                                           first_page=i,
                                                           last_page=i,
                                                           dpi=300)[0])
                cache_file.write_text(vis_text, encoding="utf-8")

            merged = gpt_merge(vis_text, raw_text)
            results.append(merged)
        else:
            results.append(
                {"merged_text": raw_text, "omissions": [], "table_json": []}
            )

    return results


# -----------------------------------------------------------------------------
# CLI entry-point (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import json as _json

    ap = argparse.ArgumentParser(description="Robust PDF extractor with GPT-4o fallback.")
    ap.add_argument("pdf", type=Path, help="Path to the input PDF file")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Write JSON lines to file")
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.is_file():
        sys.exit(f"No such PDF: {pdf_path}")

    outputs = extract_pdf_safely(pdf_path.read_bytes())

    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for page_json in outputs:
                fh.write(_json.dumps(page_json, ensure_ascii=False) + "\n")
        logger.info("Wrote %d page records to %s", len(outputs), args.out)
    else:
        print(_json.dumps(outputs, indent=2, ensure_ascii=False))
