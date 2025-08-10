import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import pytesseract

"""
Stage 1 OCR processing module.

Takes raw OCR output (from Tesseract) and restructures it into a hierarchical format: Pages → Lines → Words. 
Each line maintains a list of words (in reading order), and both lines and words retain positional metadata
(left, top, width, height). 
This structure enables consistent downstream parsing, line grouping, and phrase detection.
"""


def render_page_to_image(doc, page_num, dpi=225):
    page = doc.load_page(page_num)
    # Use DPI only (no extra matrix scaling); grayscale
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

def preprocess_image(image):
    # Gentle denoise that preserves small dots:
    # (bilateral is good; avoids smearing fine details)
    den = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    # Unsharp mask (safer than a 3x3 sharpen kernel)
    blur = cv2.GaussianBlur(den, (0, 0), 1.0)
    usm = cv2.addWeighted(den, 1.25, blur, -0.25, 0)

    # DO NOT binarize or dilate yet. Let Tesseract handle it.
    return usm


def ocr_image_with_positions(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    custom_config = r'--oem 1 --psm 6'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
    df = pd.DataFrame(data)
    df = df[(df.conf != '-1') & (df.text.str.strip() != '')].copy()
    df["text"] = df["text"].str.strip()
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    return df

def ocr_pdf_to_raw_data(pdf_path):
    """Stage 1: Convert PDF to structured OCR output with lines and words."""
    doc = fitz.open(pdf_path)
    pages_output = []

    for page_num in range(doc.page_count):
        image = render_page_to_image(doc, page_num)
        processed_img = preprocess_image(image)
        df = ocr_image_with_positions(processed_img)

        lines_output = []
        for line_num in sorted(df["line_num"].unique()):
            line_df = df[df["line_num"] == line_num].sort_values(by="left")
            words = line_df.to_dict(orient="records")
            line_text = " ".join([w["text"] for w in words])

            if words:
                line_y = int(words[0]["top"])  # Use top of first word as line y
            else:
                line_y = None

            lines_output.append({
                "line_num": int(line_num),
                "y": line_y,
                "line_text": line_text,
                "words": words
            })

        pages_output.append({
            "page_number": page_num + 1,
            "lines": lines_output,
            "image": processed_img
        })

    doc.close()

    return {
        "file_name": pdf_path.split("\\")[-1],
        "pages": pages_output
    }

