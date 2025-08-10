from ocr.helpers import OcrProfile, PreprocessSettings, TesseractSettings

OCR_SETTINGS = OcrProfile(
    name="REVOLUT",
    preprocess=PreprocessSettings(
        dpi=225,
        use_unsharp_mask=True,
        tesseract_handles_threshold=True,
    ),
    tesseract=TesseractSettings(oem=1, psm=6, lang="eng"),
)
