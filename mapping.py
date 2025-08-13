from ocr.model_settings import AIB_SETTINGS
from ocr.model_settings import BOI_SETTINGS
from ocr.model_settings import N26_SETTINGS
from ocr.model_settings import REVOLUT_SETTINGS

OCR_SETTINGS = {
    "AIB": AIB_SETTINGS,
    "BOI": BOI_SETTINGS,
    "N26": N26_SETTINGS,
    "REVOLUT": REVOLUT_SETTINGS,
}


from parsers.aib import parse_statement as aib_parser
from parsers.boi_current import parse_statement as boi_parser
from parsers.n26_current import parse_statement as n26_parser
from parsers.revolut import parse_statement as revolut_parser


BANK_PARSERS = {
    "AIB": aib_parser,
    "BOI": boi_parser,
    "N26": n26_parser,
    "REVOLUT": revolut_parser,
}



