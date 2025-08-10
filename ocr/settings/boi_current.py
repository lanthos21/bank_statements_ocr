from ocr.helpers import OcrProfile, PreprocessSettings, TesseractSettings

OCR_SETTINGS = OcrProfile(
    name="BOI",
    preprocess=PreprocessSettings(
        dpi=400,
        use_adaptive_threshold=True,
        morph_close=True,
        resize_fx=1.3, resize_fy=1.3,
        use_sharpen_kernel=True,
        enhance_contrast=True,
        emphasize_dots_dilate=True,
        tesseract_handles_threshold=False,
    ),
    tesseract=TesseractSettings(oem=1, psm=6, lang="eng"),
)
