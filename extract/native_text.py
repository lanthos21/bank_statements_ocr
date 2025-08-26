# extract/native_text.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from extract.unified_lines import merge_native_page_to_ocr_shape

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
