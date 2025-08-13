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

N26_SETTINGS = OcrProfile(
    name="N26",
    preprocess=PreprocessSettings(
        dpi=300,
        use_adaptive_threshold=False,   # ← important
        morph_close=False,              # ← avoid column bleeding
        resize_fx=1.2, resize_fy=1.2,
        use_unsharp_mask=True,          # mild clarity only
        use_sharpen_kernel=False,
        enhance_contrast=False,
        emphasize_dots_dilate=False,
        tesseract_handles_threshold=True,  # feed grayscale; no Otsu beforehand
    ),
    # psm 4 (single column, ragged right) works well on table-like pages.
    # If you still miss lines, try psm=6. Keep oem=1.
    tesseract=TesseractSettings(oem=1, psm=6, lang="eng"),
)


