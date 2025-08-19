from ocr.helper_classes import OcrProfile, PreprocessSettings, TesseractSettings

AIB_SETTINGS = OcrProfile(
    name="AIB",
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

N26_SETTINGS = OcrProfile(
    name="N26",
    preprocess=PreprocessSettings(
        dpi=200,
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

PTSB_SETTINGS = OcrProfile(
    name="PTSB",
    preprocess=PreprocessSettings(
        dpi=500,                             # bump a bit for tiny table digits
        # (If you can crop, exclude the right info panel)
        # crop_left=0.0, crop_top=0.0, crop_right=0.88, crop_bottom=1.0,  # left 88% only
        use_adaptive_threshold=False,        # feed grayscale
        tesseract_handles_threshold=True,    # let Tesseract binarize internally
        resize_fx=1.5, resize_fy=1.5,       # gentle upscale
        # resize_interpolation="cubic",        # IMPORTANT when upscaling
        use_sharpen_kernel=False,            # avoid the 3×3 kernel
        use_unsharp_mask=True,               # if your pipeline supports it
        # unsharp_amount=0.7, unsharp_radius=1.2,
        morph_close=False,                   # closing often fuses table digits
        enhance_contrast=True,
        emphasize_dots_dilate=False,         # dilation can glue characters → worse OCR
        # deskew=True,                         # if you’ve got deskew support
    ),
    # PSM 3 = full automatic page segmentation (good for mixed table + text)
    tesseract=TesseractSettings(oem=1, psm=6, lang="eng",
        extra="-c preserve_interword_spaces=1"),  # helps keep tokens separated
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

