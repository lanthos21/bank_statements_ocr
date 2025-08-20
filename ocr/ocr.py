import os
import re
import fitz, cv2, numpy as np, pandas as pd, pytesseract
from pathlib import Path
from PIL import Image, ImageEnhance

from ocr.helper_classes import PreprocessSettings, TesseractSettings, OcrProfile
from ocr.detect_currency import detect_page_currency_from_text

# --- Debug toggles ---
DEBUG_OCR = True                          # turn on/off prints
FORCE_RESCUE_PAGES: set[int] = set()      # e.g. {5, 6} to force pages 5 & 6 through rescue

def _dbg(msg: str):
    if DEBUG_OCR:
        print(msg)

# ---------------------------------
# Band rescue helpers
# ---------------------------------

def _median_height_from_df(df: pd.DataFrame) -> int:
    try:
        h = pd.to_numeric(df.get("height"), errors="coerce")
        h = h[h > 0]
        return int(float(h.median())) if len(h) else 18
    except Exception:
        return 18


def _find_large_gaps(lines_output: list[dict], min_gap_px: int, page_bottom: int | None) -> list[tuple[int,int]]:
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


def _ocr_band_to_lines(processed_img: np.ndarray, y0: int, y1: int, base_config: str) -> list[dict]:
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


def _merge_rescued_lines(lines_output: list[dict], rescued: list[dict]) -> list[dict]:
    all_lines = list(lines_output) + list(rescued)
    # drop exact duplicates by (y0,y1,text)
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


def _split_line_on_regex(line_entry: dict, regex: str) -> list[dict]:
    """
    Split a single OCR line into multiple by starting a new segment at each regex match.
    regex is applied on the *word text*, ordered left→right.
    """
    import re
    words = line_entry.get("words") or []
    if not words:
        return [line_entry]
    pat = re.compile(regex, re.IGNORECASE)
    # indices where a new segment should start (word whose text matches regex)
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

def _has_amount_like(words: list[dict], page_width: int, min_x_frac: float | None) -> bool:
    """
    True if any token looks like a money amount (supports '123.45', '1,234.56', or '2770 34').
    If min_x_frac is provided, the token center must be to the right of that page fraction.
    """
    if not words:
        return False
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if _AMOUNT_RE.search(t.replace("\u00A0"," ").replace("\u2009"," ").replace("\u202F"," ")):
            if min_x_frac is None:
                return True
            # check horizontal position
            cx = int(w.get("left",0)) + 0.5*int(w.get("width",0))
            if cx >= min_x_frac * max(1, int(page_width)):
                return True
    return False


# ---------------------------------
# Rotate photo helpers
# ---------------------------------

def _projection_score(gray: np.ndarray) -> float:
    b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - b
    proj = np.sum(inv, axis=1).astype(np.float64)  # ink per row
    return float(np.var(proj))  # straighter baselines → higher variance


def _skew_suspect(gray: np.ndarray) -> tuple[bool, float, float]:
    """
    Try several small angles and report whether any improve the projection score.
    Returns (suspect, best_angle, rel_improvement).
    """
    h, w = gray.shape[:2]
    tgt = 1600.0
    scale = min(1.0, tgt / max(h, w))
    g = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    def rot(img, ang):
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def proj_score(img):
        b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inv = 255 - b
        proj = np.sum(inv, axis=1).astype(np.float64)
        return float(np.var(proj))

    s0 = proj_score(g)

    # test multiple small angles around 0°
    angles = [-6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6]
    best_s, best_ang = s0, 0.0
    for a in angles:
        sa = proj_score(rot(g, a))
        if sa > best_s:
            best_s, best_ang = sa, float(a)

    rel = (best_s - s0) / max(s0, 1e-6)
    # threshold is small, we only need a *hint* of improvement to try deskew
    return (rel > 0.03), best_ang, rel  # 3% default trigger


def _auto_rotate_osd(gray: np.ndarray) -> np.ndarray:
    """Use Tesseract OSD to rotate upright (0/90/180/270)."""
    try:
        osd = pytesseract.image_to_osd(gray)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        angle = int(m.group(1)) if m else 0
    except Exception:
        angle = 0
    if angle == 0:
        return gray
    if angle == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


def _deskew_by_projection_search(gray: np.ndarray, max_deg: float = 6.0, trace: bool = False) -> tuple[np.ndarray, float]:
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
        if trace: _dbg(f"[deskew] step={step:.1f} best={best_ang:+.2f}° score={best_score:.0f}")

    if abs(best_ang) < 0.2:
        if trace: _dbg("[deskew] |angle| < 0.2°, skipping")
        return gray, 0.0

    M = cv2.getRotationMatrix2D((w/2, h/2), best_ang, 1.0)
    desk = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if trace: _dbg(f"[deskew] applied {best_ang:+.2f}° to full-res")
    return desk, best_ang


def _preprocess_scan_rescue(gray: np.ndarray) -> np.ndarray:
    """Better for photos/scans: de-shadow, local contrast, binarize, sharpen."""
    # de-shadow by estimating background
    bg = cv2.medianBlur(gray, 31)
    norm = cv2.normalize(gray.astype(np.float32) - bg.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    # local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    norm = clahe.apply(norm)
    # mild denoise
    norm = cv2.bilateralFilter(norm, d=5, sigmaColor=30, sigmaSpace=30)
    # adaptive threshold (Sauvola-ish via Gaussian)
    binimg = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    # unsharp
    blur = cv2.GaussianBlur(binimg, (0,0), 1.2)
    sharp = cv2.addWeighted(binimg, 1.3, blur, -0.3, 0)
    return sharp


# ---------------------------------
# Save raster
# ---------------------------------

def _safe_write_png(path: str, arr: np.ndarray) -> str:
    """Write arr as PNG reliably; fallback to PIL if OpenCV write fails or tiny file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, arr)
    try:
        too_small = (not ok) or (not os.path.exists(path)) or (os.path.getsize(path) < 1024)
    except Exception:
        too_small = True
    if too_small:
        Image.fromarray(arr).save(path, format="PNG")
    return path


# ---------------------------------
# Main OCR steps
# ---------------------------------

def render_page_to_image(doc, page_num: int, dpi: int) -> np.ndarray:
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


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
        src = gray  # feed as-is
    else:
        # mild Otsu for banks that like binarization
        src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(src, config=ts.build_config(), output_type=pytesseract.Output.DICT)
    df = pd.DataFrame(data)
    df = df[(df.conf != '-1') & (df.text.str.strip() != '')].copy()
    df["text"] = df["text"].str.strip()
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df


# ---------------------------------
# Orchestrate OCR
# ---------------------------------

def ocr_pdf_to_raw_data(pdf_path: str, profile: OcrProfile, bank_code: str | None = None) -> dict:
    """
    Rasterize + OCR each PDF page, returning a structured dict:
      {
        "file_name": <str>,
        "pages": [
          {
            "page_number": int,
            "currency": str|None,
            "currency_detect_method": str,
            "currency_confidence": float,
            "lines": [ { line_num, y, y0, y1, line_text, words:[{text,left,top,width,height,right,bottom,...}]} ],
            "raster_path": <png path>,
            "image_width": int,
            "image_height": int,
          }, ...
        ]
      }

    Notes:
    - Saves per-page processed rasters under results/ocr_rasters for later ROI OCR.
    - Optionally runs a small-angle deskew + scan-friendly rescue OCR when a page looks weak/skewed.
    - Always supports a generic band/tail "gap rescue" (profile-driven) that re-OCRs large vertical gaps
      and merges any rescued lines; stays bank-agnostic via `profile.band_rescue`.
    """
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import cv2
    import pytesseract
    import fitz

    doc = fitz.open(pdf_path)
    pages_output = []

    # Where we keep page rasters for downstream ROI re-OCR
    out_dir = os.path.join("results", "ocr_rasters")
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(pdf_path).stem

    try:
        for page_num in range(doc.page_count):
            pno = page_num + 1
            page = doc.load_page(page_num)

            # ---- Currency detection from native PDF text (no OCR) ----
            cur, cur_method, cur_conf = (None, "unknown", 0.0)
            if bank_code:
                try:
                    c, m, cf = detect_page_currency_from_text(page, bank_code)
                    cur, cur_method, cur_conf = c, m, cf
                except Exception:
                    pass

            # ---- Base raster + preprocess ----
            base = render_page_to_image(doc, page_num, profile.preprocess.dpi)
            processed = preprocess_image(base, profile.preprocess)

            # Persist processed raster so parsers can run ROI crops later
            raster_path = os.path.join(out_dir, f"{stem}_p{pno:03d}.png")
            _safe_write_png(raster_path, processed)

            # ---- Base OCR ----
            df = ocr_image_with_positions(
                processed,
                profile.tesseract,
                profile.preprocess.tesseract_handles_threshold
            )

            # Page-level OCR metrics (for debug + rescue heuristics)
            try:
                conf_ser = pd.to_numeric(df.get("conf"), errors="coerce")
                mean_conf = float(conf_ser[conf_ser >= 0].mean()) if conf_ser is not None else 0.0
                word_count = int((df.get("text") is not None) and (df.text.str.strip() != "").sum())
            except Exception:
                mean_conf, word_count = 0.0, 0
            print(f"[p{pno}] base words={word_count}, conf≈{mean_conf:.1f}")

            # ---- Decide whether to attempt a small-angle "scan rescue" OCR ----
            # Quick skew suspicion: projection variance improvement over tiny rotations
            def _projection_score(gray_img: np.ndarray) -> float:
                b = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                inv = 255 - b
                proj = np.sum(inv, axis=1).astype(np.float64)
                return float(np.var(proj))

            def _skew_suspect(gray_img: np.ndarray) -> tuple[bool, float, float]:
                h, w = gray_img.shape[:2]
                tgt = 1600.0
                scale = min(1.0, tgt / max(h, w))
                g = cv2.resize(gray_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray_img

                def rot(img, ang):
                    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
                    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                s0 = _projection_score(g)
                angles = [-6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6]
                best_s, best_ang = s0, 0.0
                for a in angles:
                    sa = _projection_score(rot(g, a))
                    if sa > best_s:
                        best_s, best_ang = sa, float(a)
                rel = (best_s - s0) / max(s0, 1e-6)
                return (rel > 0.03), best_ang, rel  # 3% improvement → suspect

            gray_for_skew = processed if processed.ndim == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            suspect, best_ang_hint, rel_impr = _skew_suspect(gray_for_skew)
            print(f"[p{pno}] skew check: suspect={bool(suspect)} (best≈{best_ang_hint:+.1f}°, {rel_impr*100:+.1f}%)")

            # Looser thresholds on tail pages (often camera photos)
            is_tail = (pno >= doc.page_count)  # last page only; use (pno >= doc.page_count-1) for last two
            low_words = word_count < (200 if is_tail else 120)
            low_conf  = mean_conf < (70.0 if is_tail else 55.0)
            skew_flag = bool(suspect) or (is_tail and rel_impr > 0.02)

            NEEDS_RESCUE = low_words or low_conf or skew_flag

            processed2 = None  # will hold scan-preprocessed raster if built

            if NEEDS_RESCUE:
                print(f"[p{pno}] rescue triggered ({', '.join(k for k,v in {'low_words':low_words,'low_conf':low_conf,'skew':skew_flag}.items() if v)})")
                base_cfg = profile.tesseract.build_config()

                # Re-render as RGB (photos fare better), then to gray
                pix_rgb = page.get_pixmap(dpi=profile.preprocess.dpi, colorspace=fitz.csRGB, alpha=False)
                rgb = np.frombuffer(pix_rgb.samples, dtype=np.uint8).reshape(pix_rgb.height, pix_rgb.width, 3)
                gray2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

                # Small-angle deskew search (±6°), traced inside helper if you enabled it there
                gray2, ang = _deskew_by_projection_search(gray2, max_deg=6.0)
                print(f"[p{pno}] deskew angle={ang:+.2f}°")

                # Scan-friendly preprocessing (de-shadow, CLAHE, adaptive thr, unsharp)
                processed2 = _preprocess_scan_rescue(gray2)

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

                print(f"[p{pno}] rescue words={word_count2} (Δ {word_count2 - word_count}), conf≈{mean_conf2:.1f} (Δ {mean_conf2 - mean_conf:.1f})")

                # Keep full rescue only if it materially improves global metrics
                if (word_count2 > word_count + 40) or (mean_conf2 > mean_conf + 8):
                    print(f"[p{pno}] rescue ACCEPTED")
                    df = df2
                    processed = processed2
                    rescue_path = os.path.join(out_dir, f"{stem}_p{pno:03d}_rescue.png")
                    _safe_write_png(rescue_path, processed2)
                else:
                    print(f"[p{pno}] rescue rejected (no material improvement)")
                    # Even if rejected, we will still use processed2 (if present) for band rescues below.

            # ---- Build base lines_output from the kept df ----
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

            # ---- Generic band/tail gap rescue (profile-driven, bank-agnostic) ----
            rescue_cfg = getattr(profile, "band_rescue", None)
            if rescue_cfg and getattr(rescue_cfg, "enabled", False) and lines_output:
                med_h = _median_height_from_df(df)
                gap_mult = float(getattr(rescue_cfg, "gap_multiplier", 1.25))
                min_gap  = max(20, int(med_h * gap_mult))
                include_tail = bool(getattr(rescue_cfg, "include_tail_gap", True))
                page_bottom = int(processed.shape[0]) - 6 if include_tail else None
                max_rescues = int(getattr(rescue_cfg, "max_rescues_per_page", 12))
                require_digit = bool(getattr(rescue_cfg, "require_digit", True))
                filter_regex  = getattr(rescue_cfg, "filter_regex", None)  # optional regex string

                # Prefer any scan-preprocessed raster for band OCR (even if full rescue was rejected)
                band_img = processed2 if processed2 is not None else processed

                gaps = _find_large_gaps(lines_output, min_gap, page_bottom)
                rescued_lines = []
                if gaps:
                    base_cfg = profile.tesseract.build_config()
                    tested = 0
                    for (gy0, gy1) in gaps:
                        if tested >= max_rescues:
                            break
                        band_lines = _ocr_band_to_lines(band_img, gy0, gy1, base_cfg)
                        # profile-driven config (add these if not already present)
                        min_amount_x_frac = getattr(rescue_cfg, "min_amount_x_frac", None)  # e.g. 0.45 for “right half”
                        stopwords_regex = getattr(rescue_cfg, "stopwords_regex", None)  # optional

                        # ... inside the gaps loop, after `band_lines` is built ...
                        # 1) split Franken-lines on repeated date tokens (if filter_regex provided)
                        if filter_regex:
                            split_lines = []
                            for bl in band_lines:
                                split_lines.extend(_split_line_on_regex(bl, filter_regex))
                            band_lines = split_lines

                        # 2) generic filters
                        kept = []
                        for bl in band_lines:
                            txt = bl.get("line_text", "") or ""
                            if stopwords_regex and re.search(stopwords_regex, txt):
                                continue
                            if require_digit and not any(ch.isdigit() for ch in txt):
                                continue
                            # require at least one amount-like token; optionally require it towards the right
                            if not _has_amount_like(bl.get("words") or [], page_width=int(processed.shape[1]),
                                                    min_x_frac=min_amount_x_frac):
                                continue
                            # optional guard: too many date tokens → likely furniture collage
                            if filter_regex and len(re.findall(filter_regex, txt, flags=re.IGNORECASE)) > 2:
                                continue
                            # optional guard: absurdly long rescued line (likely furniture)
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
                    lines_output = _merge_rescued_lines(lines_output, rescued_lines)
                    after = len(lines_output)
                    print(f"[p{pno}] band rescue merged {after - before} lines from {len(gaps)} gaps")

            # ---- Finalize page ----
            pages_output.append({
                "page_number": pno,
                "currency": cur,
                "currency_detect_method": cur_method,
                "currency_confidence": round(cur_conf, 2),
                "lines": lines_output,
                "raster_path": raster_path,
                "image_width": int(processed.shape[1]),
                "image_height": int(processed.shape[0]),
            })

    finally:
        try:
            doc.close()
        except Exception:
            pass

    return {"file_name": pdf_path.split("\\")[-1], "pages": pages_output}
