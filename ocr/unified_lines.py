# unified_lines.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from statistics import median
import math

"""
Goal
----
Make native-text extraction look like the OCR path:
- One logical "line" per row (Date + Details + Debit + Credit + Balance)
- Same shape as your OCR lines: dict(line_num, y/y0/y1, line_text, words[...], native=True/False)

Usage
-----
# 1) You already have native lines per page from your extractor:
#    native_pages: List[List[Dict]]  # pages -> lines (like your samples)
#
# 2) Merge to OCR-like shape:
#    merged_pages = merge_native_pages_to_ocr_shape(native_pages)
#
# 3) Feed `merged_pages` to your existing parser (the same one you use for OCR).

Notes
-----
- We *only* change grouping (horizontal merge by row). We do not guess headers or columns.
- We keep your fields and add/normalize where missing.
- Tolerance is derived from the page’s median word height so it adapts to different PDFs.
"""

def _norm_text(s: str) -> str:
    return " ".join((s or "").split())

def _word_bounds(w: Dict[str, Any]) -> Tuple[int, int, int, int]:
    left = int(w.get("left", 0))
    top = int(w.get("top", 0))
    width = int(w.get("width", max(0, int(w.get("right", left)) - left)))
    height = int(w.get("height", max(0, int(w.get("bottom", top)) - top)))
    right = int(w.get("right", left + width))
    bottom = int(w.get("bottom", top + height))
    return left, top, right, bottom

def _collect_words_from_native_lines(native_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten words from native 'lines' (your current structures)."""
    out: List[Dict[str, Any]] = []
    for ln in native_lines or []:
        for w in ln.get("words") or []:
            text = _norm_text(w.get("text", ""))
            if not text:
                continue
            left, top, right, bottom = _word_bounds(w)
            out.append({
                "text": text,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
                # Keep any extras if present
                "block_num": w.get("block_num", -1),
                "par_num": w.get("par_num", -1),
                "line_num": w.get("line_num", -1),
            })
    return out

def _estimate_row_tolerance(words: List[Dict[str, Any]]) -> int:
    """Choose a vertical tolerance to decide which words belong on the same row."""
    if not words:
        return 3
    heights = [max(1, int(w["height"])) for w in words]
    m = median(heights) if heights else 8
    # Allow overlap/nearby words within ~40% of a typical line height, min 2 px, max 12 px
    tol = int(round(max(2, min(12, 0.4 * m))))
    return tol

def _try_place_in_bucket(buckets: List[Dict[str, Any]],
                         word: Dict[str, Any],
                         tol: int) -> bool:
    """Place a word in an existing bucket if vertical overlap/near is sufficient."""
    w_top, w_bot = word["top"], word["bottom"]
    w_mid = (w_top + w_bot) / 2.0

    for b in buckets:
        b_top, b_bot = b["y0"], b["y1"]
        b_mid = (b_top + b_bot) / 2.0

        # Condition 1: vertical overlap with a margin
        overlap = min(w_bot, b_bot + tol) - max(w_top, b_top - tol)
        if overlap > 0:
            b["words"].append(word)
            b["y0"] = min(b["y0"], w_top)
            b["y1"] = max(b["y1"], w_bot)
            return True

        # Condition 2: midlines within tolerance (handles thin fonts)
        if abs(w_mid - b_mid) <= tol:
            b["words"].append(word)
            b["y0"] = min(b["y0"], w_top)
            b["y1"] = max(b["y1"], w_bot)
            return True

    return False

def _build_lines_from_buckets(buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn buckets into OCR-style lines, sorting words by x and joining to line_text."""
    # sort buckets by vertical position
    buckets.sort(key=lambda b: (int((b["y0"] + b["y1"]) / 2), b["y0"]))

    out_lines: List[Dict[str, Any]] = []
    for i, b in enumerate(buckets, start=1):
        words = b["words"]
        # sort by x to get reading order across columns
        words.sort(key=lambda w: (w["left"], w["top"]))

        # Reindex line-internal word numbers (like tesseract does)
        merged_words: List[Dict[str, Any]] = []
        for j, w in enumerate(words, start=1):
            # Make sure right/bottom present and consistent
            left, top, right, bottom = _word_bounds(w)
            merged_words.append({
                "text": w["text"],
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
                "right": right,
                "bottom": bottom,
                "block_num": w.get("block_num", -1),
                "par_num": w.get("par_num", -1),
                "line_num": j,  # within this new merged line
            })

        line_text = " ".join(w["text"] for w in merged_words)
        y0, y1 = b["y0"], b["y1"]
        y_mid = int(round((y0 + y1) / 2.0))

        out_lines.append({
            "line_num": i,
            "y": y_mid,
            "y0": int(y0),
            "y1": int(y1),
            "line_text": line_text,
            "words": merged_words,
            "native": True,  # this path was built from native text
        })

    return out_lines

def merge_native_page_to_ocr_shape(native_lines: List[Dict[str, Any]],
                                   row_tol_px: Optional[int] = None
                                   ) -> List[Dict[str, Any]]:
    """
    Take a single page of *native* lines (your current structure) and
    merge horizontally so each transaction row becomes one line.
    Output matches your OCR line schema.
    """
    words = _collect_words_from_native_lines(native_lines)
    if not words:
        return []

    tol = row_tol_px if isinstance(row_tol_px, int) and row_tol_px > 0 else _estimate_row_tolerance(words)
    # Make buckets (rows)
    buckets: List[Dict[str, Any]] = []
    # Sort words roughly by vertical position, then horizontal to improve locality
    words.sort(key=lambda w: (w["top"], w["left"]))

    for w in words:
        placed = _try_place_in_bucket(buckets, w, tol)
        if not placed:
            left, top, right, bottom = _word_bounds(w)
            buckets.append({"y0": top, "y1": bottom, "words": [w]})

    # Turn buckets into merged lines
    return _build_lines_from_buckets(buckets)

def merge_native_pages_to_ocr_shape(native_pages: List[List[Dict[str, Any]]],
                                    row_tol_px: Optional[int] = None
                                    ) -> List[List[Dict[str, Any]]]:
    """
    Apply `merge_native_page_to_ocr_shape` to each page.
    """
    merged: List[List[Dict[str, Any]]] = []
    for page_idx, page_lines in enumerate(native_pages or [], start=1):
        merged.append(merge_native_page_to_ocr_shape(page_lines, row_tol_px=row_tol_px))
    return merged

# ---------------------------
# Optional helpers for parity
# ---------------------------

def normalize_ocr_pages_shape(ocr_pages: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """
    Ensure OCR pages have right/bottom on every word and the expected keys.
    (No-op for most setups; safe to use on your current OCR output.)
    """
    norm_pages: List[List[Dict[str, Any]]] = []
    for page in ocr_pages or []:
        norm_lines: List[Dict[str, Any]] = []
        for ln in page or []:
            words = []
            for w in ln.get("words") or []:
                left, top, right, bottom = _word_bounds(w)
                words.append({
                    "text": _norm_text(w.get("text", "")),
                    "left": left, "top": top,
                    "width": right - left, "height": bottom - top,
                    "right": right, "bottom": bottom,
                    "block_num": w.get("block_num", -1),
                    "par_num": w.get("par_num", -1),
                    "line_num": w.get("line_num", 0),
                })
            norm_lines.append({
                "line_num": ln.get("line_num", 0),
                "y": int(ln.get("y", (ln.get("y0", 0) + ln.get("y1", 0)) // 2)),
                "y0": int(ln.get("y0", 0)),
                "y1": int(ln.get("y1", 0)),
                "line_text": _norm_text(ln.get("line_text", "")),
                "words": words,
                "native": bool(ln.get("native", False)),
            })
        norm_pages.append(norm_lines)
    return norm_pages
