#!/usr/bin/env python3
"""
Add near-invisible background watermarks to a PDF.
Each page receives a unique random token as the watermark text.

Usage:
  python invisible_watermark_pdf.py \
      --input in.pdf \
      --output out.pdf \
      --log watermark_map.csv \
      --opacity 0.03
"""

import argparse
import io
import os
import random
import string
import csv

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import portrait

# ---------- Utilities ----------

def rand_token(length: int = 10) -> str:
    alphabet = string.digits + string.ascii_lowercase
    return 'draft by xujun@ieee.org'.join(random.choice(alphabet) for _ in range(length))

def make_watermark_page_bytes(
    text: str,
    width: float,
    height: float,
    angle_deg: float,
    opacity: float = 0.03,
    font_name: str = "Helvetica",
    font_size: int = 52,
) -> bytes:
    """
    Create a single-page PDF (as bytes) with a low-alpha text watermark.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import portrait

    packet = io.BytesIO()
    w, h = portrait((width, height))
    c = canvas.Canvas(packet, pagesize=(w, h))

    # set transparency (if supported)
    try:
        c.setFillAlpha(opacity)
        c.setStrokeAlpha(opacity)
    except AttributeError:
        # fallback: no alpha (older reportlab), just light gray
        pass

    c.setFillGray(0.5)
    c.saveState()
    c.translate(w / 2.0, h / 2.0)
    c.rotate(angle_deg)
    c.setFont(font_name, font_size)

    spacing = 200  # points between repeats
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            x = dx * spacing + random.uniform(-10, 10)
            y = dy * spacing + random.uniform(-10, 10)
            c.drawCentredString(x, y, text)

    c.restoreState()
    c.showPage()
    c.save()
    return packet.getvalue()


def add_invisible_background_watermarks(
    input_pdf_path: str,
    output_pdf_path: str,
    log_csv_path: str,
    opacity: float = 0.03,
):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Record page -> token mapping
    mapping = []

    num_pages = len(reader.pages)
    for i in range(num_pages):
        page = reader.pages[i]
        # Current page size
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)

        # Generate a unique token and random angle per page
        token = rand_token(10)
        angle = random.uniform(20.0, 70.0)

        # Make watermark (one page PDF in-memory)
        wm_bytes = make_watermark_page_bytes(
            text=token,
            width=w,
            height=h,
            angle_deg=angle,
            opacity=opacity
        )

        wm_reader = PdfReader(io.BytesIO(wm_bytes))
        wm_page = wm_reader.pages[0]

        # Merge: put original content on TOP of the watermark to ensure it's a background
        # We draw original onto the watermark page, then add to writer
        wm_page.merge_page(page)  # original over background
        writer.add_page(wm_page)

        mapping.append({"page_number": i + 1, "token": token, "angle_deg": f"{angle:.2f}"})

    # Preserve existing metadata if present
    if reader.metadata:
        writer.add_metadata(reader.metadata)

    # Write out watermarked PDF
    with open(output_pdf_path, "wb") as f_out:
        writer.write(f_out)

    # Write CSV log
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f_log:
        writer_csv = csv.DictWriter(f_log, fieldnames=["page_number", "token", "angle_deg"])
        writer_csv.writeheader()
        writer_csv.writerows(mapping)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Add near-invisible background watermarks to a PDF.")
    parser.add_argument("--input", "-i", required=True, default=r"D:\Writing\llm_rag\Complete_Book.pdf", help="Path to input PDF")
    parser.add_argument("--output", "-o", required=True,default=r"D:\Writing\llm_rag\Complete_Book_mk.pdf", help="Path to output (watermarked) PDF")
    parser.add_argument("--log", "-l", default="watermark_map.csv", help="CSV path for page→token mapping")
    parser.add_argument("--opacity", "-a", type=float, default=0.00, help="Watermark opacity (0.0–1.0). Default: 0.03")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input PDF not found: {args.input}")

    if not (0.0 <= args.opacity <= 1.0):
        raise ValueError("Opacity must be between 0.0 and 1.0")

    add_invisible_background_watermarks(
        input_pdf_path=args.input,
        output_pdf_path=args.output,
        log_csv_path=args.log,
        opacity=args.opacity,
    )
    print(f"Done. Output: {args.output}\nPage→token log: {args.log}")

if __name__ == "__main__":
    main()
