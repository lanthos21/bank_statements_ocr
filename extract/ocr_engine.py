# extract/ocr_engine.py
from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pytesseract
import cv2
import fitz

from extract.helper_classes import OcrProfile
from extract.helpers import (
    ocr_image_with_positions,
    deskew_by_projection_search,
    preprocess_scan_rescue,
    find_large_gaps,
    ocr_band_to_lines,
    merge_rescued_lines,
    median_height_from_df,
)

DEBUG_OCR = True
def _dbg(msg: str):
    if DEBUG_OCR:
        print(msg)


def ocr_page_build_lines(
    page,
    page_num: int,
    processed: np.ndarray,
    profile: OcrProfile,
) -> List[Dict[str, Any]]:
    """
    OCR a single page and return OCR-shaped line dicts (identical to previous pipeline).
    Includes optional skew-detect + scan rescue + band/tail gap rescue (profile-driven).
    """
    # --- Base OCR ---
    df = ocr_image_with_positions(
        processed,
        profile.tesseract,
        profile.preprocess.tesseract_handles_threshold
    )

    # Page-level OCR metrics
    try:
        conf_ser = pd.to_numeric(df.get("conf"), errors="coerce")
        mean_conf = float(conf_ser[conf_ser >= 0].mean()) if conf_ser is not None else 0.0
        word_count = int((df.get("text") is not None) and (df.text.str.strip() != "").sum())
    except Exception:
        mean_conf, word_count = 0.0, 0

    # --- Decide small-angle "scan rescue" ---
    gray_for_skew = processed if processed.ndim == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # fast skew suspicion via projection-variance search
    suspect, best_ang, rel_impr = _skew_suspect(gray_for_skew)

    # Looser thresholds on tail pages (if you want last-page boost)
    # (keeping same behavior: last page only; adjust if needed)
    is_tail = False  # caller can modify if desired
    low_words = word_count < (200 if is_tail else 120)
    low_conf  = mean_conf < (70.0 if is_tail else 55.0)
    skew_flag = bool(suspect) or (is_tail and rel_impr > 0.02)

    needs_rescue = low_words or low_conf or skew_flag
    processed2 = None

    if needs_rescue:
        base_cfg = profile.tesseract.build_config()
        # Re-render as RGB → gray (page already given; we just use pixmap through PyMuPDF)
        pix = page.get_pixmap(dpi=profile.preprocess.dpi, colorspace=fitz.csRGB, alpha=False)  # handled by fitz
        # fitz returns RGB bytes; decode properly:
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        gray2, ang = deskew_by_projection_search(gray2, max_deg=6.0)
        processed2 = preprocess_scan_rescue(gray2)

        # Rescue OCR
        data2 = pytesseract.image_to_data(
            processed2,
            config=base_cfg if "--psm" in base_cfg else (base_cfg + " --psm 6"),
            output_type=pytesseract.Output.DICT
        )
        df2 = pd.DataFrame(data2)
        df2 = df2[(df2.conf != '-1') & (df2.text.str.strip() != '')].copy()

        mean_conf2 = 0.0
        word_count2 = 0
        if not df2.empty:
            df2["text"] = df2["text"].str.strip()
            df2["right"] = pd.to_numeric(df2["left"], errors="coerce").fillna(0).astype(int) + pd.to_numeric(df2["width"], errors="coerce").fillna(0).astype(int)
            df2["bottom"] = pd.to_numeric(df2["top"], errors="coerce").fillna(0).astype(int) + pd.to_numeric(df2["height"], errors="coerce").fillna(0).astype(int)
            conf2 = pd.to_numeric(df2.get("conf"), errors="coerce")
            mean_conf2 = float(conf2[conf2 >= 0].mean()) if conf2 is not None else 0.0
            word_count2 = int((df2.get("text") is not None) and (df2.text.str.strip() != "").sum())

        if (word_count2 > word_count + 40) or (mean_conf2 > mean_conf + 8):
            df = df2
            processed = processed2

    # --- Build base lines_output from the kept df ---
    lines_output: list[dict] = []
    if not df.empty:
        for ln in sorted(df["line_num"].unique()):
            line_df = df[df["line_num"] == ln].sort_values(by="left")
            if profile.post_line_hook is not None:
                line_df = profile.post_line_hook(line_df, processed)

            words = line_df.to_dict(orient="records")
            line_text = " ".join(w["text"] for w in words)
            y0 = int(min(w["top"] for w in words)) if words else None
            y1 = int(max(w["bottom"] for w in words)) if words else None
            y_mid = int(words[0]["top"]) if words else None

            lines_output.append({
                "line_num": int(ln),
                "y": y_mid,
                "y0": y0,
                "y1": y1,
                "line_text": line_text,
                "words": words
            })

    # --- Generic band/tail gap rescue (profile-driven) ---
    rescue_cfg = getattr(profile, "band_rescue", None)
    if rescue_cfg and getattr(rescue_cfg, "enabled", False) and lines_output:
        med_h = median_height_from_df(df)
        gap_mult = float(getattr(rescue_cfg, "gap_multiplier", 1.25))
        min_gap  = max(20, int(med_h * gap_mult))
        include_tail = bool(getattr(rescue_cfg, "include_tail_gap", True))
        page_bottom = int(processed.shape[0]) - 6 if include_tail else None
        max_rescues = int(getattr(rescue_cfg, "max_rescues_per_page", 12))
        require_digit = bool(getattr(rescue_cfg, "require_digit", True))
        filter_regex  = getattr(rescue_cfg, "filter_regex", None)  # optional regex string
        min_amount_x_frac = getattr(rescue_cfg, "min_amount_x_frac", None)
        stopwords_regex = getattr(rescue_cfg, "stopwords_regex", None)

        band_img = processed2 if processed2 is not None else processed
        gaps = find_large_gaps(lines_output, min_gap, page_bottom)
        rescued_lines = []
        if gaps:
            base_cfg = profile.tesseract.build_config()
            tested = 0
            for (gy0, gy1) in gaps:
                if tested >= max_rescues:
                    break
                band_lines = ocr_band_to_lines(band_img, gy0, gy1, base_cfg)

                # split on repeated tokens if configured
                if filter_regex:
                    from .helpers import split_line_on_regex
                    split_lines = []
                    for bl in band_lines:
                        split_lines.extend(split_line_on_regex(bl, filter_regex))
                    band_lines = split_lines

                kept = []
                for bl in band_lines:
                    txt = bl.get("line_text", "") or ""
                    if stopwords_regex and __import__("re").search(stopwords_regex, txt):
                        continue
                    if require_digit and not any(ch.isdigit() for ch in txt):
                        continue
                    # require at least one amount-like token; optionally require it towards the right
                    from .helpers import has_amount_like
                    if not has_amount_like(bl.get("words") or [], page_width=int(processed.shape[1]),
                                           min_x_frac=min_amount_x_frac):
                        continue
                    if filter_regex and len(__import__("re").findall(filter_regex, txt, flags=__import__("re").IGNORECASE)) > 2:
                        continue
                    if len(txt) > int(getattr(rescue_cfg, "max_rescued_text_len", 160)):
                        continue
                    kept.append(bl)
                rescued_lines.extend(kept)

        if rescued_lines:
            # Run post_line_hook on rescued too (for consistency)
            if profile.post_line_hook is not None:
                for bl in rescued_lines:
                    ldf = pd.DataFrame(bl["words"]).sort_values(by="left")
                    ldf = profile.post_line_hook(ldf, band_img)
                    bl["words"] = ldf.to_dict(orient="records")
                    bl["line_text"] = " ".join(w["text"] for w in bl["words"])
                    if bl["words"]:
                        bl["y"]  = int(bl["words"][0]["top"])
                        bl["y0"] = int(min(w["top"] for w in bl["words"]))
                        bl["y1"] = int(max(w["bottom"] for w in bl["words"]))

            before = len(lines_output)
            lines_output = merge_rescued_lines(lines_output, rescued_lines)
            after = len(lines_output)
            _dbg(f"[p{page_num}] band rescue merged {after - before} lines")

    # Final monotonic sort for safety
    lines_output.sort(key=lambda r: (int(r.get("y0", 10**9)), int(r.get("y", 10**9))))
    return lines_output


# --- local private: quick skew probe (same idea as in your existing code) ---
def _skew_suspect(gray: np.ndarray) -> tuple[bool, float, float]:
    h, w = gray.shape[:2]
    tgt = 1600.0
    scale = min(1.0, tgt / max(h, w))
    g = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    def rot(img, ang):
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    def proj_score(img):
        b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - b
        proj = np.sum(inv, axis=1).astype(np.float64)
        return float(np.var(proj))

    s0 = proj_score(g)
    angles = [-6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6]
    best_s, best_ang = s0, 0.0
    for a in angles:
        sa = proj_score(rot(g, a))
        if sa > best_s:
            best_s, best_ang = sa, float(a)
    rel = (best_s - s0) / max(s0, 1e-6)
    return (rel > 0.03), best_ang, rel
