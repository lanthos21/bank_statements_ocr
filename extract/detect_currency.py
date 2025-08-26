import re
import fitz
from constants import CURRENCIES


def detect_page_currency_from_text(page, bank_code: str):
    if bank_code == "BOI":
        return "EUR", "rule", 1.0
    if bank_code == "N26":
        return "EUR", "rule", 1.0

    w, h = page.rect.width, page.rect.height
    header_rect = fitz.Rect(0, 0, w, h * 0.18)  # top 18% full width
    header_txt = page.get_text("text", clip=header_rect) or ""
    lines = [ln.strip() for ln in header_txt.splitlines() if ln.strip()]

    # 1️⃣ Prefer a currency code that appears on same line as 'Statement'
    for line in lines:
        if "statement" in line.lower():
            m = re.search(r"\b([A-Z]{3})\b", line)
            if m:
                cur = m.group(1).upper()
                if cur in CURRENCIES:
                    return cur, "header-statement", 0.98

    # 2️⃣ Fallback: first known code in the header block
    for line in lines:
        m = re.search(r"\b([A-Z]{3})\b", line)
        if m:
            cur = m.group(1).upper()
            if cur in CURRENCIES:
                return cur, "header-lines", 0.95

    return None, "unknown", 0.0


