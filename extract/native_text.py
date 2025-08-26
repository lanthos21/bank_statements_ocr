# extract/native_text.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def native_page_to_ocr_shape_lines(
    page,
    img_w: int,
    img_h: int,
    post_line_hook=None,
    processed_img=None,
) -> List[Dict[str, Any]]:
    """
    Use native PDF text and emit OCR-shaped lines:
      [{line_num,y,y0,y1,line_text,words:[{text,left,top,width,height,right,bottom,...}], native=True}]
    Coords are scaled to the raster size (img_w,img_h) for parser parity.
    """
    try:
        # words: [x0,y0,x1,y1,"text", block_no, line_no, word_no]
        words = page.get_text("words") or []
    except Exception:
        words = []

    if not words:
        return []

    # Scale PDF points -> raster pixels
    rect = page.rect
    fx = (img_w / float(rect.width)) if rect.width else 1.0
    fy = (img_h / float(rect.height)) if rect.height else 1.0

    # group by (block_no, line_no) to get "native lines"
    from collections import defaultdict
    buckets = defaultdict(list)
    for x0, y0, x1, y1, txt, bno, lno, wno in words:
        t = (txt or "").strip()
        if not t:
            continue
        left   = int(round(x0 * fx)); right  = int(round(x1 * fx))
        top    = int(round(y0 * fy)); bottom = int(round(y1 * fy))
        buckets[(int(bno), int(lno))].append({
            "text": t,
            "left": left,
            "top": top,
            "width": max(0, right - left),
            "height": max(0, bottom - top),
            "right": right,
            "bottom": bottom,
            "block_num": int(bno),
            "par_num": -1,
            "line_num": int(wno),
        })

    native_lines = []
    for _, wlist in buckets.items():
        if not wlist:
            continue
        wlist.sort(key=lambda w: (w["left"], w["top"]))
        y0 = min(w["top"] for w in wlist)
        y1 = max(w["bottom"] for w in wlist)
        native_lines.append({
            "line_num": -1,
            "y": int(wlist[0]["top"]),
            "y0": int(y0),
            "y1": int(y1),
            "line_text": " ".join(w["text"] for w in wlist),
            "words": wlist,
        })

    merged = merge_native_page_to_ocr_shape(native_lines)

    # Keep numbering stable and mark origin
    for i, ln in enumerate(merged, start=1):
        ln["line_num"] = i
        ln["native"] = True

    # Apply same post_line_hook (parity with OCR path)
    if post_line_hook is not None and processed_img is not None:
        new = []
        for ln in merged:
            df_line = pd.DataFrame(ln["words"]).sort_values(by="left")
            df_line = post_line_hook(df_line, processed_img)
            words2 = df_line.to_dict(orient="records")
            if words2:
                ln["words"] = words2
                ln["line_text"] = " ".join(w["text"] for w in words2)
                ln["y"]  = int(words2[0]["top"])
                ln["y0"] = int(min(w["top"] for w in words2))
                ln["y1"] = int(max(w["bottom"] for w in words2))
                new.append(ln)
        merged = new

    return merged


def merge_native_page_to_ocr_shape(native_lines: List[Dict],
                                   y_eps_px: int | None = None,
                                   wrap_merge: bool = True) -> List[Dict]:
    if not native_lines:
        return []

    heights = [max(1, int(ln["y1"]) - int(ln["y0"])) for ln in native_lines if ln.get("y0") is not None and ln.get("y1") is not None]
    med_h = int(np.median(heights)) if heights else 18
    row_eps = int(max(6, 0.55 * med_h)) if y_eps_px is None else int(y_eps_px)

    def ymid(ln): return (int(ln["y0"]) + int(ln["y1"])) // 2
    lines = sorted(native_lines, key=lambda ln: (int(ln.get("y0", 10**9)), int(ln.get("y", 10**9))))

    clusters: list[list[Dict]] = []
    centers: list[int] = []

    for ln in lines:
        ym = ymid(ln)
        if not clusters:
            clusters.append([ln]); centers.append(ym); continue
        diffs = [abs(ym - c) for c in centers]
        k = int(np.argmin(diffs))
        if diffs[k] <= row_eps:
            clusters[k].append(ln)
            centers[k] = int(np.mean([(l["y0"] + l["y1"]) / 2 for l in clusters[k]]))
        else:
            clusters.append([ln]); centers.append(ym)

    out: List[Dict] = []
    for cl in clusters:
        words = []
        for ln in cl: words.extend(ln.get("words") or [])
        if not words: continue
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

    out.sort(key=lambda ln: (int(ln["y0"]), int(ln["y"])))
    for ln in out:
        t = (ln.get("line_text") or "")
        if ("IBAN" in t or "BIC" in t) and len(t) > 120:
            ln["hint_long_finref"] = True
    return out
