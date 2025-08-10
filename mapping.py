from ocr.settings.boi_current import OCR_SETTINGS as BOI_SETTINGS
from ocr.settings.revolut import OCR_SETTINGS as REVOLUT_SETTINGS

OCR_SETTINGS = {
    "BOI": BOI_SETTINGS,
    "REVOLUT": REVOLUT_SETTINGS,
}


from parsers.boi_current import parse_boi_statement
from parsers.revolut import parse_revolut_statement

BANK_PARSERS = {
    "BOI": parse_boi_statement,
    "REVOLUT": parse_revolut_statement,
}



