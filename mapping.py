from ocr.model_settings import BOI_SETTINGS
from ocr.model_settings import REVOLUT_SETTINGS
from ocr.model_settings import N26_SETTINGS

OCR_SETTINGS = {
    "BOI": BOI_SETTINGS,
    "REVOLUT": REVOLUT_SETTINGS,
    "N26": N26_SETTINGS,
}


from parsers.boi_current import parse_statement as boi_parser
from parsers.revolut import parse_statement as revolut_parser
from parsers.n26_current import parse_statement as n26_parser


BANK_PARSERS = {
    "BOI": boi_parser,
    "REVOLUT": revolut_parser,
    "N26": n26_parser,
}



