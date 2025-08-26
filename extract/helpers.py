# extract/helpers.py
from __future__ import annotations
import os, re
from typing import List, Dict, Any, Tuple
from pathlib import Path

import fitz
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance

from .helper_classes import PreprocessSettings, TesseractSettings

# -------- I/O & rendering --------

def render_page_to_image(doc, page_num: int, dpi: int) -> np.ndarray:
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

def safe_write_png(path: str, arr: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, arr)
    try:
        too_small = (not ok) or (not os.path.exists(path)) or (os.path.getsize(path) < 1024)
    except Exception:
        too_small = True
    if too_small:
        Image.fromarray(arr).save(path, format="PNG")
    return path

def preprocess_image(img: np.ndarray, s: PreprocessSettings) -> np.ndarray:
    out = img
    if s.use_adaptive_threshold:
        out = cv2.adaptiveThreshold(
            out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            s.adaptive_block_size, s.adaptive_C
        )
    if s.morph_close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, s.morph_kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if s.resize_fx != 1.0 or s.resize_fy != 1.0:
        out = cv2.resize(out, None, fx=s.resize_fx, fy=s.resize_fy, interpolation=s.resize_interpolation)
    if s.use_unsharp_mask:
        blur = cv2.GaussianBlur(out, (0, 0), s.usm_radius_sigma)
        out = cv2.addWeighted(out, s.usm_amount, blur, s.usm_subtract, 0)
    if s.use_sharpen_kernel:
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        out = cv2.filter2D(out, -1, k)
    if s.enhance_contrast:
        pil = Image.fromarray(out)
        out = np.array(ImageEnhance.Contrast(pil).enhance(s.contrast_factor))
    if s.emphasize_dots_dilate:
        out = cv2.dilate(out, np.ones(s.dilate_kernel, np.uint8), iterations=s.dilate_iterations)
    return out

def ocr_image_with_positions(img: np.ndarray, ts: TesseractSettings, tesseract_handles_threshold: bool) -> pd.DataFrame:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if tesseract_handles_threshold:
        src = gray
    else:
        src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(src, config=ts.build_config(), output_type=pytesseract.Output.DICT)
    df = pd.DataFrame(data)
    df = df[(df.conf != '-1') & (df.text.str.strip() != '')].copy()
    df["text"] = df["text"].str.strip()
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df

# -------- scan rescue helpers --------

def deskew_by_projection_search(gray: np.ndarray, max_deg: float = 6.0) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    tgt = 1600.0
    scale = min(1.0, tgt / max(h, w))
    g = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    def rotate(img, ang):
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    def score(img):
        b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - b
        proj = np.sum(inv, axis=1).astype(np.float64)
        return float(np.var(proj))

    best_ang, best_score = 0.0, -1e9
    center, radius = 0.0, max_deg
    for step in (1.5, 0.5, 0.2):
        angs = np.arange(center - radius, center + radius + 1e-6, step)
        for ang in angs:
            s = score(rotate(g, ang))
            if s > best_score:
                best_score, best_ang = s, float(ang)
        center, radius = best_ang, max(step*1.8, 0.4)

    if abs(best_ang) < 0.2:
        return gray, 0.0

    M = cv2.getRotationMatrix2D((w/2, h/2), best_ang, 1.0)
    desk = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return desk, best_ang

def preprocess_scan_rescue(gray: np.ndarray) -> np.ndarray:
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.normalize(gray.astype(np.float32) - bg.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    norm = clahe.apply(norm)
    norm = cv2.bilateralFilter(norm, d=5, sigmaColor=30, sigmaSpace=30)
    binimg = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    blur = cv2.GaussianBlur(binimg, (0,0), 1.2)
    sharp = cv2.addWeighted(binimg, 1.3, blur, -0.3, 0)
    return sharp

# -------- band rescue helpers --------

def median_height_from_df(df: pd.DataFrame) -> int:
    try:
        h = pd.to_numeric(df.get("height"), errors="coerce")
        h = h[h > 0]
        return int(float(h.median())) if len(h) else 18
    except Exception:
        return 18

def find_large_gaps(lines_output: list[dict], min_gap_px: int, page_bottom: int | None) -> list[tuple[int,int]]:
    rows = [ln for ln in lines_output if ln.get("y0") is not None and ln.get("y1") is not None]
    rows.sort(key=lambda r: int(r["y0"]))
    gaps = []
    for a, b in zip(rows, rows[1:]):
        gap = int(b["y0"]) - int(a["y1"])
        if gap >= min_gap_px:
            gaps.append((int(a["y1"]) + 1, int(b["y0"]) - 1))
    if page_bottom is not None and rows:
        tail_gap = int(page_bottom) - int(rows[-1]["y1"])
        if tail_gap >= min_gap_px:
            gaps.append((int(rows[-1]["y1"]) + 1, int(page_bottom)))
    return gaps

def ocr_band_to_lines(processed_img: np.ndarray, y0: int, y1: int, base_config: str) -> list[dict]:
    H, W = processed_img.shape[:2]
    ya = max(0, min(y0, y1)); yb = min(H, max(y0, y1))
    if yb - ya < 8:
        return []
    roi = processed_img[ya:yb, 0:W]
    band = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cfg = base_config
    if "--psm" not in cfg:
        cfg += " --psm 6"
    if "preserve_interword_spaces" not in cfg:
        cfg += " -c preserve_interword_spaces=1"

    data = pytesseract.image_to_data(band, config=cfg, output_type=pytesseract.Output.DICT)
    df2 = pd.DataFrame(data)
    if df2.empty:
        return []
    df2 = df2[(df2.conf != '-1') & (df2.text.str.strip() != '')].copy()
    if df2.empty:
        return []

    df2["text"]   = df2["text"].str.strip()
    df2["top"]    = pd.to_numeric(df2["top"], errors="coerce").fillna(0).astype(int) + ya
    df2["left"]   = pd.to_numeric(df2["left"], errors="coerce").fillna(0).astype(int)
    df2["width"]  = pd.to_numeric(df2["width"], errors="coerce").fillna(0).astype(int)
    df2["height"] = pd.to_numeric(df2["height"], errors="coerce").fillna(0).astype(int)
    df2["right"]  = df2["left"] + df2["width"]
    df2["bottom"] = df2["top"]  + df2["height"]

    out = []
    for ln in sorted(df2["line_num"].unique()):
        ld = df2[df2["line_num"] == ln].sort_values(by="left")
        words = ld.to_dict(orient="records")
        if not words:
            continue
        line_text = " ".join(w["text"] for w in words)
        y0b = int(min(w["top"] for w in words))
        y1b = int(max(w["bottom"] for w in words))
        out.append({
            "line_num": -1,
            "y": int(words[0]["top"]),
            "y0": y0b,
            "y1": y1b,
            "line_text": line_text,
            "words": words,
            "rescued": True,
        })
    return out

def merge_rescued_lines(lines_output: list[dict], rescued: list[dict]) -> list[dict]:
    all_lines = list(lines_output) + list(rescued)
    seen = set()
    uniq = []
    for ln in all_lines:
        key = (ln.get("y0"), ln.get("y1"), ln.get("line_text"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ln)
    uniq.sort(key=lambda r: (r.get("y0", 10**9), r.get("y", 10**9)))
    for i, ln in enumerate(uniq, start=1):
        ln["line_num"] = i
    return uniq

def split_line_on_regex(line_entry: dict, regex: str) -> list[dict]:
    import re
    words = line_entry.get("words") or []
    if not words:
        return [line_entry]
    pat = re.compile(regex, re.IGNORECASE)
    starts = [i for i, w in enumerate(words) if pat.search((w.get("text") or ""))]
    if len(starts) <= 1:
        return [line_entry]
    segments = []
    for si, idx in enumerate(starts):
        jdx = starts[si + 1] if si + 1 < len(starts) else len(words)
        seg_words = words[idx:jdx]
        if not seg_words:
            continue
        y0 = int(min(w["top"] for w in seg_words))
        y1 = int(max(w["bottom"] for w in seg_words))
        seg = {
            "line_num": -1,
            "y": int(seg_words[0]["top"]),
            "y0": y0,
            "y1": y1,
            "words": seg_words,
            "line_text": " ".join(w["text"] for w in seg_words),
            "rescued": True,
        }
        segments.append(seg)
    return segments

_AMOUNT_RE = re.compile(r"(?<!\w)(?:[-+]?\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})|\d+\s\d{2})(?!\w)")
def has_amount_like(words: list[dict], page_width: int, min_x_frac: float | None) -> bool:
    if not words:
        return False
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if _AMOUNT_RE.search(t.replace("\u00A0"," ").replace("\u2009"," ").replace("\u202F"," ")):
            if min_x_frac is None:
                return True
            cx = int(w.get("left",0)) + 0.5*int(w.get("width",0))
            if cx >= min_x_frac * max(1, int(page_width)):
                return True
    return False
