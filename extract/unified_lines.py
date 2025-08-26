# extract/unified_lines.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

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
