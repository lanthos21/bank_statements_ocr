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


def render_page_to_image(doc, page_num, dpi=400):
    pix = doc.load_page(page_num).get_pixmap(
        matrix=fitz.Matrix(1, 1),
        dpi=dpi,
        colorspace=fitz.csGRAY
    )
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

def preprocess_image(image):
    # 1. Adaptive thresholding (good for scanned documents)
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 2. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. Resize up (helps Tesseract read small text/symbols better)
    resized = cv2.resize(cleaned, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR)

    # ✅ 4. Apply sharpening filter
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

    # 5. Contrast enhancement using PIL
    pil_image = Image.fromarray(sharpened)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(1.5)

    #return np.array(enhanced)
    # ✅ 6. Emphasize dots (after sharpening and contrast)
    emphasized = cv2.dilate(np.array(enhanced), np.ones((1, 1), np.uint8), iterations=1)

    return emphasized

def ocr_image_with_positions(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    custom_config = r'--oem 3 --psm 6'
    # custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,€abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:-'
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
            "lines": lines_output
        })

    doc.close()

    return {
        "file_name": pdf_path.split("\\")[-1],
        "pages": pages_output
    }

