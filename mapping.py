from parsers.boi_current import parse_boi_statement
from parsers.revolut import parse_revolut_statement

BANK_PARSERS = {
    "BOI": parse_boi_statement,
    "REVOLUT": parse_revolut_statement,
}

from ocr.settings.boi_current import PROFILE as BOI_PROFILE
from ocr.settings.revolut import PROFILE as REVOLUT_PROFILE

OCR_SETTINGS = {
    "BOI": BOI_PROFILE,
    "REVOLUT": REVOLUT_PROFILE,
    # AIB_CODE: AIB_PROFILE,
}

