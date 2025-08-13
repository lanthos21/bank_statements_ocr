
# --- Detect bank provider ---
BANK_PATTERNS = {
    "AIB": [
        r"\bAllied Irish Bank\b", r"\bAIB\b"
    ],
    "N26": [
        r"\bNTSB"
    ],
    "REVOLUT": [
        r"\bRevolut\b", r"Revolut Bank UAB",
    ],
    "BOI": [
        r"\bBank of Ireland\b", r"\bBOFI\b"
    ],

}


# --- Revolut currencies ---
CURRENCIES = {
    "GBP", "USD", "AED", "AUD", "BGN", "CAD", "CHF", "CZK", "DKK", "EUR",
    "HKD", "HUF", "ILS", "ISK", "JPY", "MXN", "NOK", "NZD", "PLN", "QAR",
    "RON", "RSD", "SAR", "SEK", "SGD", "THB", "TRY", "ZAR", "KRW", "COP",
    "PHP", "INR", "CLP"
}


