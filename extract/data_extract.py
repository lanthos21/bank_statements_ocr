from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional, Dict, Any
import fitz
from contextlib import suppress

from extract.helper_classes import OcrProfile
from extract.detect_currency import detect_page_currency_from_text
from extract.helpers import render_page_to_image, preprocess_image, safe_write_png
from extract.native_text import native_page_to_ocr_shape_lines
from extract.ocr_engine import ocr_page_build_lines

DEBUG_EXTRACT = True
def _dbg(msg: str):
    if DEBUG_EXTRACT:
        print(msg)

_AMOUNT_RE = re.compile(r"(?<!\w)(?:[-+]?\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})|\d+\s\d{2})(?!\w)")

def _native_is_sufficient(lines: list[dict]) -> bool:
    if not lines:
        return False
    for ln in lines:
        for w in (ln.get("words") or []):
            t = (w.get("text") or "").replace("\u00A0"," ").replace("\u2009"," ").replace("\u202F"," ").strip()
            if _AMOUNT_RE.search(t):
                return True
    return len(lines) >= 5

def extract_pdf_to_raw_data(
    pdf_path: str,
    profile: OcrProfile,
    bank_code: Optional[str] = None,
    strategy: Optional[str] = "auto",
) -> Dict[str, Any]:
    strategy = (strategy or "auto").lower()
    if strategy not in ("auto", "native", "ocr"):
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'auto', 'native', or 'ocr'.")

    out_pages: list[dict] = []
    out_dir = os.path.join("results", "ocr_rasters")
    stem = Path(pdf_path).stem

    # NEW: simple counters
    native_pages = 0
    ocr_pages = 0

    # Use a context manager to avoid try/finally + broad except
    with fitz.open(pdf_path) as doc:
        for page_idx in range(doc.page_count):
            pno = page_idx + 1
            page = doc.load_page(page_idx)

            # currency detection
            cur, cur_method, cur_conf = (None, "unknown", 0.0)
            if bank_code:
                # catch only unexpected currency-detection errors
                with suppress(Exception):
                    cur, cur_method, cur_conf = detect_page_currency_from_text(page, bank_code)

            base = render_page_to_image(doc, page_idx, profile.preprocess.dpi)
            processed = preprocess_image(base, profile.preprocess)
            W, H = int(processed.shape[1]), int(processed.shape[0])

            raster_path_str = os.path.join(out_dir, f"{stem}_p{pno:03d}.png")

            run_native = strategy in ("auto", "native")
            run_ocr    = strategy in ("auto", "ocr")

            lines: list[dict] = []
            native_ok = False
            text_source = "none"

            if run_native:
                lines = native_page_to_ocr_shape_lines(
                    page,
                    img_w=W,
                    img_h=H,
                    post_line_hook=profile.post_line_hook,
                    processed_img=processed,
                )
                native_ok = _native_is_sufficient(lines)

            if native_ok and strategy != "ocr":
                native_pages += 1
                text_source = "native"
                out_pages.append({
                    "page_number": pno,
                    "currency": cur,
                    "currency_detect_method": cur_method,
                    "currency_confidence": round(cur_conf, 2),
                    "lines": lines,
                    "raster_path": None,
                    "image_width": W,
                    "image_height": H,
                    "text_source": text_source,
                })
                continue

            # OCR fallback / forced
            if run_ocr and (strategy == "ocr" or not native_ok):
                os.makedirs(out_dir, exist_ok=True)
                safe_write_png(raster_path_str, processed)

                lines = ocr_page_build_lines(
                    page=page,
                    page_num=pno,
                    processed=processed,
                    profile=profile,
                )
                ocr_pages += 1
                text_source = "ocr"

                out_pages.append({
                    "page_number": pno,
                    "currency": cur,
                    "currency_detect_method": cur_method,
                    "currency_confidence": round(cur_conf, 2),
                    "lines": lines,
                    "raster_path": raster_path_str,
                    "image_width": W,
                    "image_height": H,
                    "text_source": text_source,
                })
                continue

            # native-only but weak/empty
            out_pages.append({
                "page_number": pno,
                "currency": cur,
                "currency_detect_method": cur_method,
                "currency_confidence": round(cur_conf, 2),
                "lines": [],
                "raster_path": None,
                "image_width": W,
                "image_height": H,
                "text_source": text_source,  # "none"
            })

    # NEW: simple usage summary (no more grayed-out warning)
    page_usage = {
        "native_text_pages": native_pages,
        "ocr_pages": ocr_pages,
        "total_pages_in_pdf": len(out_pages),
    }

    return {
        "file_name": Path(pdf_path).name,
        "pages": out_pages,
        "meta": {"page_usage": page_usage},
    }
