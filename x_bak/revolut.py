import re
import math
import pandas as pd
from utils import parse_currency, parse_date
from lang import HEADER_SYNONYMS


def detect_column_positions(lines: list[dict]) -> tuple[dict, int, int]:

    def match_header(text: str, keywords: list[str]) -> bool:
        return any(k in text for k in keywords)

    for idx, line in enumerate(lines):
        line_text = " ".join(w["text"] for w in line["words"] if "text" in w).lower()

        # Must have at least date + description to be a header
        if match_header(line_text, HEADER_SYNONYMS["date"]) and \
           match_header(line_text, HEADER_SYNONYMS["description"]) and \
           (match_header(line_text, HEADER_SYNONYMS["money_out"]) or
            match_header(line_text, HEADER_SYNONYMS["money_in"])):

            from typing import Optional

            positions: dict[str, Optional[int]] = {
                "date": None,
                "description": None,
                "money_out": None,
                "money_in": None,
                "balance": None
            }

            for i, word in enumerate(line["words"]):
                text = word.get("text", "").lower()
                x = word.get("x", word.get("left"))  # support either key

                if any(text == k or k in text for k in HEADER_SYNONYMS["date"]):
                    positions["date"] = int(x)
                elif any(text == k or k in text for k in HEADER_SYNONYMS["description"]):
                    positions["description"] = int(x)
                elif any(text == k or k in text for k in HEADER_SYNONYMS["balance"]):
                    positions["balance"] = int(x)

                # Handle two-word headers like "Money out" or "Dinero saliente"
                if i + 1 < len(line["words"]):
                    combined = f"{text} {line['words'][i+1].get('text','').lower()}"
                    if any(combined == k for k in HEADER_SYNONYMS["money_out"]):
                        positions["money_out"] = int(x)
                    elif any(combined == k for k in HEADER_SYNONYMS["money_in"]):
                        positions["money_in"] = int(x)

            return ({
                "date": positions["date"],
                "desc": positions["description"],
                "out":  positions["money_out"],
                "in":   positions["money_in"],
                "bal":  positions["balance"],
            }, idx, len(lines))

    raise ValueError("❌ Could not find transaction header row.")


def categorise_amount_by_left_edge(x1, header_positions, margin=100):
    for category in ["out", "in", "bal"]:
        pos = header_positions.get(category)
        if pos is not None and abs(x1 - pos) <= margin:
            return category
    return "unknown"


def statement_date_range(pages: list[dict]) -> tuple[str | None, str | None]:
    pattern = r"from (\d{1,2} \w+ \d{4}) to (\d{1,2} \w+ \d{4})"
    for page in pages:
        for line in page["lines"]:
            match = re.search(pattern, line["line_text"], re.IGNORECASE)
            if match:
                return match.group(1), match.group(2)
    return None, None


def transaction_description(words, desc_x, first_amount_x, tol=8):
    """
    Extracts description text between the Description column and the first amount column,
    with a small tolerance so first words aren't chopped and amounts don't sneak in.
    """
    if desc_x is None or first_amount_x is None:
        return ""

    parts = []
    for w in sorted(words, key=lambda ww: ww.get("left", ww.get("x", 0))):
        x_left = w.get("left", w.get("x", 0))
        if x_left < desc_x - tol:   # allow small left bleed
            continue
        if x_left >= first_amount_x - tol:  # stop a little early
            break
        parts.append(w.get("text", ""))

    return " ".join(t for t in parts if t).strip()


def parse_transactions(
    pages: list[dict],
    iban: str | None = None
) -> list[dict]:
    """
    Assumes each page may carry:
      page["currency"] -> "EUR"/"GBP"/...
      page["currency_detect_method"], page["currency_confidence"] (optional)
    Falls back to IBAN hint, then to previous page, then 'EUR'.
    Resets running balance on currency change.
    """

    def infer_page_currency(page: dict, prev_cur: str | None) -> str:
        # 1) what OCR/text detection put on the page
        cur = page.get("currency")
        if isinstance(cur, str) and len(cur) == 3:
            return cur.upper()
        # 2) IBAN hint (e.g. GB... => GBP, IE... => EUR)
        if iban:
            u = iban.upper()
            if u.startswith("GB"): return "GBP"
            if u.startswith("IE"): return "EUR"
        # 3) previous page’s currency
        if prev_cur: return prev_cur
        # 4) fallback (safe default)
        return "EUR"

    all_transactions: list[dict] = []

    # Totals by currency (credits/debits)
    totals_in: dict[str, float] = {}
    totals_out: dict[str, float] = {}

    running_balance_cents: int | None = None
    current_currency: str | None = None

    # Date range helpers (as you already had)
    start_date_str, end_date_str = statement_date_range(pages)
    initial_value_date = parse_date(start_date_str) if start_date_str else None

    last_seen_date = initial_value_date

    for page_num, page in enumerate(pages):
        # Decide currency for this page and reset running balance if section changes
        page_currency = infer_page_currency(page, current_currency)

        if current_currency != page_currency:
            # New currency section begins -> reset running balance
            running_balance_cents = None
            current_currency = page_currency
            # ensure totals dicts have keys
            totals_in.setdefault(current_currency, 0.0)
            totals_out.setdefault(current_currency, 0.0)

        lines = page.get("lines", [])
        if not lines:
            continue

        # Detect column positions for this page
        try:
            header_positions, start_idx, end_idx = detect_column_positions(lines)
        except ValueError:
            # Couldn't find table header on this page
            continue

        # Iterate the table rows in this page
        for line in lines[start_idx + 1 : end_idx]:
            df = pd.DataFrame(line["words"])
            if df.empty:
                continue

            # Normalize expected columns
            df.rename(columns={"word": "text", "x": "left"}, inplace=True, errors="ignore")
            df["line_text"] = line["line_text"]

            # Parse amounts and classify columns by x-position
            df["amount"] = df["text"].apply(parse_currency)
            df["category"] = df["left"].apply(
                lambda x: categorise_amount_by_left_edge(x, header_positions)
            )

            # Require a date on the line (no fallback)
            date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", line["line_text"])
            if not date_match:
                continue
            parsed_date = parse_date(date_match.group(0))
            if not parsed_date:
                continue
            transaction_date = parsed_date
            last_seen_date = transaction_date

            # Amounts by category
            in_series = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit_amount = in_series.iloc[0] if not in_series.empty else None
            debit_amount = out_series.iloc[0] if not out_series.empty else None
            balance = bal_series.iloc[0] if not bal_series.empty else None

            # Must have date, balance, and either in or out
            if (credit_amount is None and debit_amount is None) or balance is None:
                continue

            # after header_positions, inside the row loop:
            amount_cols = [v for k, v in header_positions.items() if k in ("in", "out", "bal") and v is not None]
            first_amount_x = min(amount_cols) if amount_cols else None

            clean_desc = transaction_description(
                line["words"],
                header_positions.get("desc"),
                first_amount_x
            )

            # Normalize numeric values
            credit_amount = 0.0 if (credit_amount is None or math.isnan(credit_amount)) else float(credit_amount)
            debit_amount  = 0.0 if (debit_amount  is None or math.isnan(debit_amount))  else float(debit_amount)

            # Accumulate per-currency totals
            totals_in.setdefault(current_currency, 0.0)
            totals_out.setdefault(current_currency, 0.0)
            totals_in[current_currency]  += credit_amount
            totals_out[current_currency] += debit_amount

            # Running balance logic (reset happens on currency change above)
            credit_amount_cents = int(round(credit_amount * 100))
            debit_amount_cents  = int(round(debit_amount  * 100))

            if running_balance_cents is None:
                running_balance_cents = int(round(balance * 100))
            else:
                running_balance_cents += credit_amount_cents - debit_amount_cents

            calculated_balance = round(running_balance_cents / 100.0, 2)

            # Optional: report discrepancy
            if not math.isclose(calculated_balance, float(balance), abs_tol=0.01):
                discrepancy = round(float(balance) - calculated_balance, 2)
                print("\n⚠️ BALANCE DISCREPANCY DETECTED")
                print(f"📄 Page: {page_num + 1} | 💱 Currency: {current_currency}")
                print(f"📆 Date: {transaction_date}")
                print(f"📝 Description: {clean_desc}")
                print(f"💸 Credit: {credit_amount:.2f} | Debit: {debit_amount:.2f}")
                print(f"📊 Calculated Balance: {calculated_balance:.2f}")
                print(f"📄 Statement Balance:  {float(balance):.2f}")
                print(f"🧮 Discrepancy: {discrepancy:+.2f}")
                print(f"🔤 OCR Line: {line['line_text']}")

            # Build transaction entry
            amt_value = credit_amount if credit_amount > 0 else debit_amount
            all_transactions.append({
                "transactions_date": transaction_date,
                "transaction_type": "credit" if credit_amount > 0 else "debit",
                "description": clean_desc,
                "amount": {
                    "value": amt_value,
                    "currency": current_currency,
                },
                "balance_after_statement": {
                    "value": float(balance),
                    "currency": current_currency,
                },
                "balance_after_calculated": {
                    "value": calculated_balance,
                    "currency": current_currency,
                },
            })

    # Summary per currency
    print(f"\n📄 TOTAL REVOLUT TRANSACTIONS: {len(all_transactions)}")
    for cur in sorted(set(list(totals_in.keys()) + list(totals_out.keys()))):
        _in  = totals_in.get(cur, 0.0)
        _out = totals_out.get(cur, 0.0)
        print(f"  • {cur}: IN {_in:.2f} | OUT {_out:.2f}")

    return all_transactions


def parse_statement(raw_ocr, client="Unknown", account_type="Unknown"):
    full_text = "\n".join("\n".join(line["line_text"] for line in page["lines"]) for page in raw_ocr["pages"])

    iban_match = re.search(r"IBAN\s+([A-Z]{2}\d{2}[A-Z0-9]{11,30})", full_text)
    iban = iban_match.group(1).strip() if iban_match else None
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    start_date_str, end_date_str = statement_date_range(raw_ocr["pages"])
    start_date = parse_date(start_date_str) if start_date_str else None
    end_date = parse_date(end_date_str) if end_date_str else None

    transactions = parse_transactions(raw_ocr["pages"], iban=iban)

    return {
        "client": client,
        "file_name": raw_ocr["file_name"],
        "account_holder": None,
        "institution": "Revolut",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "opening_balance": None,
        "closing_balance_statement": transactions[-1]["balance_after_statement"] if transactions else None,
        "closing_balance_calculated": transactions[-1]["balance_after_calculated"] if transactions else None,
        "transactions": transactions,
    }
