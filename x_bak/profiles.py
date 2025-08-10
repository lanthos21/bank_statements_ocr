from ocr.ocr import PreprocessSettings, TesseractSettings, OcrProfile
import pandas as pd
import numpy as np

def _noop_post(df: pd.DataFrame, _img: np.ndarray) -> pd.DataFrame:
    return df

BOI_PROFILE = OcrProfile(
    name="BOI",
    preprocess=PreprocessSettings(
        dpi=400,
        use_adaptive_threshold=True, adaptive_block_size=11, adaptive_C=2,
        morph_close=True, morph_kernel=(2,2),
        resize_fx=1.3, resize_fy=1.3,
        use_unsharp_mask=False, use_sharpen_kernel=True,
        enhance_contrast=True, contrast_factor=1.5,
        emphasize_dots_dilate=True,
        tesseract_handles_threshold=False,  # we already binarize
    ),
    tesseract=TesseractSettings(
        oem=1, psm=6, lang="eng"
    ),
    post_line_hook=_noop_post
)

REVOLUT_PROFILE = OcrProfile(
    name="REVOLUT",
    preprocess=PreprocessSettings(
        dpi=225,
        # gentle path: no pre-threshold, let Tesseract decide
        use_adaptive_threshold=False,
        morph_close=False,
        resize_fx=1.0, resize_fy=1.0,
        use_unsharp_mask=True, usm_amount=1.25, usm_radius_sigma=1.0, usm_subtract=-0.25,
        use_sharpen_kernel=False,
        enhance_contrast=False,
        emphasize_dots_dilate=False,
        tesseract_handles_threshold=True,
    ),
    tesseract=TesseractSettings(
        oem=1, psm=6, lang="eng"
    ),
    post_line_hook=_noop_post
)

PROFILES = {
    "BOI": BOI_PROFILE,
    "REVOLUT": REVOLUT_PROFILE,
    # Add more banks here as you learn their quirks
}
