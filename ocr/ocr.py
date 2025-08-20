import os
from pathlib import Path

import fitz, cv2, numpy as np, pandas as pd, pytesseract
from PIL import Image, ImageEnhance
from ocr.helper_classes import PreprocessSettings, TesseractSettings, OcrProfile
from ocr.detect_currency import detect_page_currency_from_text

import re

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

def _deskew_small_angle(gray: np.ndarray) -> np.ndarray:
    """Estimate small skew via Hough lines; rotate a few degrees if needed."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 180)
    if lines is None:
        return gray
    angles = []
    for rho, theta in lines[:,0]:
        # ignore near-vertical lines
        deg = (theta * 180.0 / np.pi)
        if 20 < deg % 180 < 160:
            angles.append(deg - 90)
    if not angles:
        return gray
    angle = np.median(angles)
    if abs(angle) < 0.5 or abs(angle) > 7:  # keep it conservative
        return gray
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

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


def ocr_pdf_to_raw_data(pdf_path: str, profile: OcrProfile, bank_code: str | None = None) -> dict:
    doc = fitz.open(pdf_path)
    pages_output = []

    # NEW: where we keep the processed page rasters so a parser can do ROI re-OCR
    out_dir = os.path.join("results", "ocr_rasters")
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(pdf_path).stem

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # 🔹 Currency detection from PDF text (fast, no OCR)
            cur, cur_method, cur_conf = (None, "unknown", 0.0)
            if bank_code:
                c, m, cf = detect_page_currency_from_text(page, bank_code)
                cur, cur_method, cur_conf = c, m, cf

            # Raster + preprocess (your existing settings)
            base = render_page_to_image(doc, page_num, profile.preprocess.dpi)
            processed = preprocess_image(base, profile.preprocess)

            # NEW: persist processed raster so we can crop ROIs later
            raster_path = os.path.join(out_dir, f"{stem}_p{page_num+1:03d}.png")
            _safe_write_png(raster_path, processed)

            df = ocr_image_with_positions(
                processed,
                profile.tesseract,
                profile.preprocess.tesseract_handles_threshold
            )

            # after df is computed:
            try:
                conf = pd.to_numeric(df.get("conf"), errors="coerce")
                mean_conf = float(conf[conf >= 0].mean()) if conf is not None else 0.0
                word_count = int((df.get("text") is not None) and (df.text.str.strip() != "").sum())
            except Exception:
                mean_conf, word_count = 0.0, 0

            # Heuristic: if page is poor, try scan-friendly re-OCR
            NEEDS_RESCUE = (word_count < 120) or (mean_conf < 55.0)

            if NEEDS_RESCUE:
                # 1) re-render in RGB (photo scans tend to like this), then to gray
                pix_rgb = page.get_pixmap(dpi=profile.preprocess.dpi, colorspace=fitz.csRGB, alpha=False)
                rgb = np.frombuffer(pix_rgb.samples, dtype=np.uint8).reshape(pix_rgb.height, pix_rgb.width, 3)
                gray2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

                # 2) auto-rotate & deskew a bit
                gray2 = _auto_rotate_osd(gray2)
                gray2 = _deskew_small_angle(gray2)

                # 3) scan rescue preprocess
                processed2 = _preprocess_scan_rescue(gray2)

                # 4) OCR with threshold handled by us (tesseract_handles_threshold=False)
                df2 = pytesseract.image_to_data(
                    processed2,
                    config=profile.tesseract.build_config().replace("--psm 6", "--psm 6"),
                    output_type=pytesseract.Output.DICT
                )
                df2 = pd.DataFrame(df2)
                df2 = df2[(df2.conf != '-1') & (df2.text.str.strip() != '')].copy()
                df2["text"] = df2["text"].str.strip()
                df2["right"] = df2["left"] + df2["width"]
                df2["bottom"] = df2["top"] + df2["height"]

                # 5) choose the better page (more words or higher mean confidence)
                conf2 = pd.to_numeric(df2.get("conf"), errors="coerce")
                mean_conf2 = float(conf2[conf2 >= 0].mean()) if conf2 is not None else 0.0
                word_count2 = int((df2.get("text") is not None) and (df2.text.str.strip() != "").sum())

                if (word_count2 > word_count + 40) or (mean_conf2 > mean_conf + 8):
                    df = df2
                    processed = processed2
                    # save the rescue raster too
                    rescue_path = os.path.join(out_dir, f"{stem}_p{page_num + 1:03d}_rescue.png")
                    _safe_write_png(rescue_path, processed2)

            lines_output = []
            for ln in sorted(df["line_num"].unique()):
                line_df = df[df["line_num"] == ln].sort_values(by="left")
                if profile.post_line_hook is not None:
                    line_df = profile.post_line_hook(line_df, processed)

                words = line_df.to_dict(orient="records")
                line_text = " ".join(w["text"] for w in words)
                line_y = int(words[0]["top"]) if words else None

                # NEW: keep tight vertical bounds for this line (for ROI crops)
                if len(words) > 0:
                    y0 = int(min(w["top"] for w in words))
                    y1 = int(max(w["bottom"] for w in words))
                else:
                    y0 = y1 = None

                lines_output.append({
                    "line_num": int(ln),
                    "y": line_y,
                    "y0": y0,                 # NEW
                    "y1": y1,                 # NEW
                    "line_text": line_text,
                    "words": words
                })

            pages_output.append({
                "page_number": page_num + 1,
                "currency": cur,
                "currency_detect_method": cur_method,
                "currency_confidence": round(cur_conf, 2),
                "lines": lines_output,
                # NEW: help the parser crop ROIs
                "raster_path": raster_path,
                "image_width": int(processed.shape[1]),
                "image_height": int(processed.shape[0]),
            })
    finally:
        doc.close()

    return {"file_name": pdf_path.split("\\")[-1], "pages": pages_output}
