from ocr.model_settings import BOI_SETTINGS
from ocr.model_settings import REVOLUT_SETTINGS

OCR_SETTINGS = {
    "BOI": BOI_SETTINGS,
    "REVOLUT": REVOLUT_SETTINGS,
}


from parsers.boi_current import parse_statement as boi_parser
from parsers.revolut import parse_statement as revolut_parser

BANK_PARSERS = {
    "BOI": boi_parser,
    "REVOLUT": revolut_parser,
}



