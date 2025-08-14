import fitz, cv2, numpy as np, pandas as pd, pytesseract
from PIL import Image, ImageEnhance
from ocr.helper_classes import PreprocessSettings, TesseractSettings, OcrProfile
from ocr.detect_currency import detect_page_currency_from_text

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
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # 🔹 Currency detection from PDF text (fast, no OCR)
            cur, cur_method, cur_conf = (None, "unknown", 0.0)
            if bank_code:
                c, m, cf = detect_page_currency_from_text(page, bank_code)
                cur, cur_method, cur_conf = c, m, cf

            # print(f"cur, cur_method, cur_conf = {c}, {m}, {cf}")

            # Continue with your existing raster + OCR (if you still need OCR)
            base = render_page_to_image(doc, page_num, profile.preprocess.dpi)
            processed = preprocess_image(base, profile.preprocess)
            df = ocr_image_with_positions(
                processed,
                profile.tesseract,
                profile.preprocess.tesseract_handles_threshold
            )

            lines_output = []
            for ln in sorted(df["line_num"].unique()):
                line_df = df[df["line_num"] == ln].sort_values(by="left")
                if profile.post_line_hook is not None:
                    line_df = profile.post_line_hook(line_df, processed)

                words = line_df.to_dict(orient="records")
                line_text = " ".join(w["text"] for w in words)
                line_y = int(words[0]["top"]) if words else None

                lines_output.append({
                    "line_num": int(ln),
                    "y": line_y,
                    "line_text": line_text,
                    "words": words
                })

            pages_output.append({
                "page_number": page_num + 1,
                "currency": cur,
                "currency_detect_method": cur_method,
                "currency_confidence": round(cur_conf, 2),
                "lines": lines_output
            })
    finally:
        doc.close()

    return {"file_name": pdf_path.split("\\")[-1], "pages": pages_output}
