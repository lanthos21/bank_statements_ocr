
# --- Detect bank provider ---
BANK_PATTERNS = {
    "REVOLUT": [
        r"\bRevolut\b", r"Revolut Bank UAB",
    ],
    "BOI": [
        r"\bBank of Ireland\b", r"\bBOFI\b"
    ],
    # add others as needed...
}



# revolut currencies
# GBP, USD, AED, AUD, BGN, CAD, CHF, CZK, DKK, EUR, HKD, HUF, ILS, ISK, JPY, MXN, NOK, NZD, PLN, QAR, RON, RSD, SAR, SEK, SGD, THB, TRY, ZAR, KRW, COP, PHP, INR, CLP
