from ocr.model_settings import BOI_SETTINGS
from ocr.model_settings import REVOLUT_SETTINGS

OCR_SETTINGS = {
    "BOI": BOI_SETTINGS,
    "REVOLUT": REVOLUT_SETTINGS,
}


from parsers.boi_current import parse_statement
from parsers.revolut import parse_statement

BANK_PARSERS = {
    "BOI": parse_statement,
    "REVOLUT": parse_statement,
}



