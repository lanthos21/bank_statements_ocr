from lang import MONTHS_MAP
import re
import shutil, time, stat
from pathlib import Path

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
        if strip_currency and value and not value[0].isdigit():
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


def parse_dateOLD(date_str: str) -> str | None:
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

from datetime import datetime
import unicodedata


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def parse_date(date_str: str) -> str | None:
    """
    Parse date strings like:
    '25 Jul 2023', '12 dic 2024', '01 ene 2025', '10 oct 2025', '15 déc 2024'
    Returns 'YYYY-MM-DD' or None if not recognised.
    """
    if not date_str:
        return None
    parts = date_str.strip().split()
    if len(parts) != 3:
        return None

    day, month_str, year = parts
    month_key = strip_accents(month_str).lower()
    month_num = MONTHS_MAP.get(month_key)
    if not month_num:
        return None

    try:
        return datetime(int(year), month_num, int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None

from datetime import datetime

def date_variants(iso_date: str) -> list[str]:
    """Return possible printed formats for an ISO date (YYYY-MM-DD)."""
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d")
    except Exception:
        return []
    return [
        d.strftime("%d %b %Y"),   # 10 May 2024
        d.strftime("%d %B %Y"),   # 10 May 2024 (long month)
        d.strftime("%d/%m/%Y"),   # 10/05/2024
        d.strftime("%d-%m-%Y"),   # 10-05-2024
    ]


def nuke_dir(path: Path, tries: int = 3, delay: float = 0.25) -> None:
    """
    Remove a directory tree, tolerating Windows file-lock weirdness.
    Retries a few times after chmod-ing read-only files.
    """
    if not path.exists():
        return

    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
        except Exception:
            pass
        try:
            func(p)
        except Exception:
            pass

    for i in range(tries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except Exception:
            if i == tries - 1:
                print(f"⚠️  Failed to delete {path} after {tries} tries.")
                return
            time.sleep(delay)
