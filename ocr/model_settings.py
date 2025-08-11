from ocr.helper_classes import OcrProfile, PreprocessSettings, TesseractSettings

BOI_SETTINGS = OcrProfile(
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

REVOLUT_SETTINGS = OcrProfile(
    name="REVOLUT",
    preprocess=PreprocessSettings(
        dpi=225,
        use_unsharp_mask=True,
        tesseract_handles_threshold=True,
    ),
    tesseract=TesseractSettings(oem=1, psm=6, lang="eng"),
)