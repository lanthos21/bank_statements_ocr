from dataclasses import dataclass
from typing import Optional, Callable
import cv2, numpy as np, pandas as pd

@dataclass
class PreprocessSettings:
    # rendering
    dpi: int = 300

    # steps (enable/disable)
    use_adaptive_threshold: bool = False
    adaptive_block_size: int = 11
    adaptive_C: int = 2

    morph_close: bool = False
    morph_kernel: tuple[int,int] = (2,2)

    resize_fx: float = 1.0
    resize_fy: float = 1.0
    resize_interpolation: int = cv2.INTER_LINEAR

    use_unsharp_mask: bool = False
    usm_amount: float = 1.25     # weight of original
    usm_radius_sigma: float = 1.0 # Gaussian sigma
    usm_subtract: float = -0.25  # weight of blurred

    use_sharpen_kernel: bool = False  # usually False if using USM
    emphasize_dots_dilate: bool = False
    dilate_kernel: tuple[int,int] = (1,1)
    dilate_iterations: int = 1

    enhance_contrast: bool = False
    contrast_factor: float = 1.5

    # Let Tesseract do its own thresholding? (feed grayscale directly)
    tesseract_handles_threshold: bool = True


@dataclass
class TesseractSettings:
    oem: int = 1
    psm: int = 6
    lang: str = "eng"
    whitelist: Optional[str] = None
    numeric_mode: bool = False
    disable_dicts: bool = False
    extra: str = ""  # any extra flags

    def build_config(self) -> str:
        cfg = f"--oem {self.oem} --psm {self.psm} -l {self.lang}"
        if self.whitelist:
            cfg += f" -c tessedit_char_whitelist={self.whitelist}"
        if self.numeric_mode:
            cfg += " -c classify_bln_numeric_mode=1"
        if self.disable_dicts:
            cfg += " -c load_system_dawg=0 -c load_freq_dawg=0"
        if self.extra:
            cfg += f" {self.extra}"
        return cfg


@dataclass
class OcrProfile:
    name: str
    preprocess: PreprocessSettings
    tesseract: TesseractSettings
    # Optional hooks if a bank needs custom tweaks after OCR
    post_line_hook: Optional[Callable[[pd.DataFrame, np.ndarray], pd.DataFrame]] = None
    # Optional: a second, focused pass (e.g., amounts column); your parser can also do this.