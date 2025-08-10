from datetime import datetime
import re

def parse_currency(value: str, strip_currency: bool = True) -> float | None:
    """
    Parses a currency string into a float.

    Args:
        value (str): Currency string (e.g. '€1,234.56', '1,234.56').
        strip_currency (bool): If True, removes the first character assuming it's a currency symbol.

    Returns:
        float | None: Parsed numeric value, or None if parsing fails.
    """
    try:
        if not value or not isinstance(value, str):
            return None

        # Step 0: Optionally strip leading currency symbols (€, £, $)
        if strip_currency:
            value = value[1:]

        # Step 1: Keep only digits, dots, commas, minus
        cleaned = re.sub(r"[^\d.,\-]", "", value)

        # Step 2: Replace commas with dots
        cleaned = cleaned.replace(",", ".")

        # Step 3: If more than one dot, treat last dot as decimal separator
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = "".join(parts[:-1]) + "." + parts[-1]

        return float(cleaned)
    except ValueError:
        return None


def parse_date(date_str: str) -> str | None:
    """
    Convert a date string like '25 Jul 2023' or '25 July 2023' into ISO format (YYYY-MM-DD).
    Tries both abbreviated and full month names.
    """
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None

