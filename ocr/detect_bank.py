import re
from pathlib import Path
import fitz
import numpy as np
import cv2
import pytesseract
from constants import BANK_PATTERNS

def _match_bank(text: str) -> tuple[str|None, float]:
    if not text:
        return None, 0.0
    text_low = text
    best = (None, 0.0)
    for bank, pats in BANK_PATTERNS.items():
        hits = sum(1 for p in pats if re.search(p, text_low, flags=re.I))
        if hits:
            # simple confidence: matched/total + small bonus for ≥2 hits
            conf = hits / len(pats) + (0.15 if hits >= 2 else 0.0)
            if conf > best[1]:
                best = (bank, conf)
    return best

def detect_bank_provider(pdf_path: str) -> tuple[str|None, float, str]:
    """
    Returns (bank_code, confidence, method) where method is 'text', 'meta', or 'ocr'.
    Uses a context manager to avoid double-closing the document.
    """
    # Open once, auto-close on any return / exception
    with fitz.open(pdf_path) as doc:
        # 0) Empty / corrupt guard
        if getattr(doc, "page_count", 0) == 0:
            return None, 0.0, "unknown"

        # 1) Try real PDF text (first 2 pages)
        raw = []
        for i in range(min(2, doc.page_count)):
            try:
                raw.append(doc.load_page(i).get_text("text"))
            except Exception:
                pass
        bank, conf = _match_bank("\n".join(raw))
        if bank:
            return bank, conf, "text"

        # 2) Try metadata
        meta_blob = " ".join([str(v) for v in (doc.metadata or {}).values() if v])
        bank, conf = _match_bank(meta_blob)
        if bank:
            return bank, conf, "meta"

        # 3) Quick OCR skim of top-right band on page 1
        try:
            p = doc.load_page(0)
            pix = p.get_pixmap(dpi=240, colorspace=fitz.csGRAY, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

            img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
            h, w = img.shape[:2]
            top_band = img[0:int(h*0.18), int(w*0.55):w]  # top-right band

            band = cv2.threshold(top_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cfg = "--oem 1 --psm 11 -l eng"
            data = pytesseract.image_to_string(band, config=cfg)
            bank, conf = _match_bank(data)
            if bank:
                return bank, conf, "ocr"

        except Exception:
            # Ignore OCR skim errors and fall through
            pass

    # Auto-closed here by the context manager
    return None, 0.0, "unknown"
