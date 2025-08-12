# boi_current.py

import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

from utils import parse_currency, parse_date


# ----------------------------
# Header / column detection (BOI)
# ----------------------------

def detect_column_positions(lines: list[dict]) -> tuple[dict, int, int] | None:
    """
    Locate BOI headers like 'Payments - in', 'Payments - out', 'Balance'.
    Returns (header_positions, header_start_idx, len(lines)) or None if not found.

    NOTE: BOI amounts are typically RIGHT-JUSTIFIED; we store right edges for
    in/out/bal columns.
    """
    header_positions = {}
    header_start_idx = None

    def find_phrase_x_right(line, phrase: str) -> int | None:
        words = line.get("words", [])
        parts = phrase.lower().split()
        for i in range(len(words) - len(parts) + 1):
            if all(parts[j] in (words[i + j].get("text", "").lower()) for j in range(len(parts))):
                w_last = words[i + len(parts) - 1]
                left = int(w_last.get("left", 0))
                width = int(w_last.get("width", 0))
                return left + width  # RIGHT edge
        return None

    for idx, line in enumerate(lines):
        if not line.get("words"):
            continue

        right_out = find_phrase_x_right(line, "Payments - out")
        right_in  = find_phrase_x_right(line, "Payments - in")
        right_bal = None
        for w in line.get("words", []):
            t = (w.get("text") or "").lower()
            if "balance" in t:
                right_bal = int(w.get("left", 0)) + int(w.get("width", 0))
                break

        if right_out and right_in and right_bal:
            header_positions = {"out": right_out, "in": right_in, "bal": right_bal}
            header_start_idx = idx
            break

    if header_start_idx is None:
        return None

    return header_positions, header_start_idx, len(lines)


def categorise_amount_by_right_edge(x_right: int, header_positions: dict, margin: int = 80) -> str:
    """
    Classify token into 'out' | 'in' | 'bal' by proximity to the column RIGHT edges.
    """
    x_right = int(x_right)
    best = ("unknown", margin + 1)
    for k in ("out", "in", "bal"):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs(x_right - int(pos))
        if d < best[1]:
            best = (k, d)
    return best[0] if best[1] <= margin else "unknown"


# ----------------------------
# Optional helpers (BOI)
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = line.get("line_text", "")
            if "IBAN" in txt.upper():
                after = txt.upper().split("IBAN", 1)[-1]
                cleaned = re.sub(r"\s+", "", after)
                m = iban_pattern.search(cleaned)
                if m:
                    return m.group(1)
    return None


def extract_opening_balance_and_start_date(pages: list[dict]) -> tuple[float | None, str | None]:
    """
    If a 'Balance forward' line exists, use its Balance value and date as the opening & start.
    Falls back to (None, None) silently.
    """
    for page in pages:
        header_result = detect_column_positions(page.get("lines", []))
        if not header_result:
            continue
        header_positions, _, _ = header_result

        for line in page.get("lines", []):
            if "balance forward" in (line.get("line_text", "").lower()):
                df = pd.DataFrame(line.get("words", []))
                if df.empty:
                    continue
                df["right"] = df["left"] + df["width"]
                df["amount"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))
                df["category"] = df["right"].apply(lambda x: categorise_amount_by_right_edge(x, header_positions))
                bal_series = df[df["category"] == "bal"]["amount"]
                balance = round(bal_series.iloc[0], 2) if not bal_series.empty else None

                date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", line.get("line_text", ""))
                start_date = parse_date(date_match.group(0)) if date_match else None
                return balance, start_date
    return None, None


# ----------------------------
# Transactions (BOI)
# ----------------------------

def parse_transactions(
    pages: list[dict],
    iban: str | None = None
) -> list[dict]:
    """
    Parse all BOI transaction rows across pages. Returns a flat list of transactions.
    Each row includes balance if present. We keep statement order via a 'seq' field.
    """
    all_transactions: List[dict] = []
    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0

    for page_num, page in enumerate(pages):
        lines = page.get("lines", []) or []
        header = detect_column_positions(lines)
        if not header:
            continue

        header_positions, start_idx, end_idx = header

        # keep the last date seen on this page/section
        last_seen_date: Optional[str] = None

        for line in lines[start_idx + 1 : end_idx]:
            words = line.get("words", []) or []
            if not words:
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue

            df["right"] = df["left"] + df["width"]
            df["line_text"] = line.get("line_text", "")

            # Only consider tokens that look somewhat numeric to speed things
            # (still apply parse_currency for robustness)
            df["amount"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))
            df["category"] = df["right"].apply(lambda x: categorise_amount_by_right_edge(x, header_positions))

            # --- NEW: carry-forward date ---
            m = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", df["line_text"].iloc[0])
            if m:
                parsed = parse_date(m.group(0))
                if parsed:
                    last_seen_date = parsed
            transaction_date = last_seen_date
            if transaction_date is None:
                # we still haven't encountered a date yet; skip until we do
                continue
            # --- end NEW ---

            in_series  = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit = float(in_series.iloc[0])  if not in_series.empty  and pd.notna(in_series.iloc[0])  else 0.0
            debit  = float(out_series.iloc[0]) if not out_series.empty and pd.notna(out_series.iloc[0]) else 0.0

            if credit == 0.0 and debit == 0.0:
                # Skip non-transaction rows (e.g., just a balance)
                continue

            stmt_balance = float(bal_series.iloc[0]) if not bal_series.empty and pd.notna(bal_series.iloc[0]) else None

            # description: drop the date only if present
            clean_desc = df["line_text"].iloc[0]
            if m:
                clean_desc = clean_desc.replace(m.group(0), "").strip()

            all_transactions.append({
                "seq": seq,
                "transactions_date": transaction_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": {
                    "value": credit if credit > 0 else debit,
                    "currency": current_currency,
                },
                "balance_after_statement": None if stmt_balance is None else {
                    "value": stmt_balance,
                    "currency": current_currency,
                },
                # BOI: we won’t maintain a running calc here (can be misleading with pockets etc.)
                "balance_after_calculated": None,
            })
            seq += 1

    return all_transactions


# ----------------------------
# Build multi-currency section (BOI)
# ----------------------------

def _group_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in transactions:
        cur = t.get("amount", {}).get("currency")
        if cur:
            buckets[cur].append(t)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))  # preserve statement order
    return buckets


def _derive_open_close_from_rows(txs: List[dict]) -> tuple[float | None, float | None]:
    """
    Opening from rows: reverse the first transaction against its statement balance (if present).
    Closing from rows: last statement balance (if present).
    """
    if not txs:
        return None, None

    first = txs[0]
    last  = txs[-1]

    first_bal = (first.get("balance_after_statement") or {}).get("value")
    first_amt = first.get("amount", {}).get("value")
    first_typ = first.get("transaction_type")

    opening = None
    if first_bal is not None and first_amt is not None and first_typ in ("credit", "debit"):
        if first_typ == "credit":
            opening = round(float(first_bal) - float(first_amt), 2)
        else:
            opening = round(float(first_bal) + float(first_amt), 2)

    closing_stmt = (last.get("balance_after_statement") or {}).get("value")
    closing_stmt = None if closing_stmt is None else float(closing_stmt)

    return opening, closing_stmt


def _build_currency_sections_from_rows(buckets: Dict[str, List[dict]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cur in sorted(buckets.keys()):
        txs = buckets[cur]

        money_in  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)

        opening_from_rows, closing_stmt_from_rows = _derive_open_close_from_rows(txs)

        # Calculate a closing from rows if we have opening
        closing_calc = None
        if opening_from_rows is not None:
            closing_calc = round(float(opening_from_rows) + money_in - money_out, 2)

        out[cur] = {
            "opening_balance": None if opening_from_rows is None else {"value": opening_from_rows, "currency": cur},
            "money_in_total":  {"value": money_in,  "currency": cur},
            "money_out_total": {"value": money_out, "currency": cur},
            "closing_balance_statement": None if closing_stmt_from_rows is None else {"value": closing_stmt_from_rows, "currency": cur},
            "closing_balance_calculated": None if closing_calc is None else {"value": closing_calc, "currency": cur},
            "transactions": txs,
        }

    return out


# ----------------------------
# Public entrypoint (BOI)
# ----------------------------

def parse_statement(raw_ocr, client="Unknown", account_type="Unknown"):
    """
    Parse a BOI statement into the same multi-currency structure used by Revolut.
    BOI usually has a single currency and no summary table, so we derive opening/closing
    from the transaction rows themselves (or 'Balance forward' if present).
    """
    pages = raw_ocr.get("pages", []) or []

    # IBAN & currency guess
    full_text = "\n".join("\n".join(line.get("line_text", "") for line in page.get("lines", [])) for page in pages)
    iban = extract_iban(pages)
    currency_hint = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Transactions (flat)
    transactions = parse_transactions(pages, iban=iban)

    # Group by currency (likely just one)
    buckets = _group_by_currency(transactions)

    # Build per-currency sections purely from rows (no “Total a b c d” on BOI)
    currencies = _build_currency_sections_from_rows(buckets)

    # Statement dates from transactions
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        # Try a fallback start date from a 'Balance forward' line if we can
        _, start_date = extract_opening_balance_and_start_date(pages)
        end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,                 # add if you later parse it
        "institution": "Bank of Ireland",
        "account_type": account_type,
        "iban": iban,
        "bic": None,                            # add if you later parse it
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,               # ← aligned with Revolut structure
    }
