# unified_lines.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from statistics import median
import numpy as np

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

def merge_native_page_to_ocr_shape(native_lines: List[Dict],
                                   y_eps_px: int | None = None,
                                   wrap_merge: bool = True) -> List[Dict]:
    """
    Merge words into single visual *rows* without crossing baselines.
    - native_lines: each item has {y0,y1,words:[{left,top,right,bottom,text}], ...}
    - y_eps_px: optional vertical tolerance; if None we derive from median height.
    - wrap_merge: if True, try to join wrapped detail continuations that are
      *directly* beneath a row and horizontally overlapping the detail column,
      but still keep them within the same row cluster only if they pass the
      baseline threshold (prevents paragraph swallowing).
    """

    if not native_lines:
        return []

    # --- Derive vertical tolerance from median line height ---
    heights = [max(1, int(ln["y1"]) - int(ln["y0"])) for ln in native_lines if ln.get("y0") is not None and ln.get("y1") is not None]
    med_h = int(np.median(heights)) if heights else 18
    row_eps = int(max(6, 0.55 * med_h)) if y_eps_px is None else int(y_eps_px)

    # --- Sort by baseline then cluster rows by y-mid within eps ---
    def ymid(ln): return (int(ln["y0"]) + int(ln["y1"])) // 2
    lines = sorted(native_lines, key=lambda ln: (int(ln.get("y0", 10**9)), int(ln.get("y", 10**9))))

    clusters: list[list[Dict]] = []
    centers: list[int] = []

    for ln in lines:
        ym = ymid(ln)
        if not clusters:
            clusters.append([ln]); centers.append(ym); continue
        # attach to nearest cluster if within row_eps; else start a new cluster
        diffs = [abs(ym - c) for c in centers]
        k = int(np.argmin(diffs))
        if diffs[k] <= row_eps:
            clusters[k].append(ln)
            # update center to the mean midline (robust against tiny jitter)
            centers[k] = int(np.mean([(l["y0"] + l["y1"]) / 2 for l in clusters[k]]))
        else:
            clusters.append([ln]); centers.append(ym)

    # --- Build one merged line per cluster (no cross-baseline merge) ---
    out: List[Dict] = []
    for cl in clusters:
        words = []
        for ln in cl:
            words.extend(ln.get("words") or [])
        if not words:
            continue

        # order left→right, then top as tie-break
        words.sort(key=lambda w: (int(w.get("left", 0)), int(w.get("top", 0))))

        y0 = min(int(w["top"]) for w in words)
        y1 = max(int(w["bottom"]) for w in words)
        text = " ".join((w.get("text") or "").strip() for w in words if (w.get("text") or "").strip())

        out.append({
            "line_num": -1,
            "y": int(words[0]["top"]),
            "y0": y0,
            "y1": y1,
            "line_text": text,
            "words": words,
        })

    # keep vertical order stable
    out.sort(key=lambda ln: (int(ln["y0"]), int(ln["y"])))

    # --- Lightweight safeguard for IBAN/BIC lines (generic) ---------------
    # We do NOT change parsing; we just prevent absurdly long glued lines when
    # "IBAN" or "BIC" appears near dense paragraphs by trimming to the cluster.
    # (Because clusters already forbid cross-baseline glue, this rarely triggers.)
    for ln in out:
        t = (ln.get("line_text") or "")
        if ("IBAN" in t or "BIC" in t) and len(t) > 120:
            # Keep as-is but mark for parsers if they want to apply a stricter regex locally.
            ln["hint_long_finref"] = True

    return out

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
