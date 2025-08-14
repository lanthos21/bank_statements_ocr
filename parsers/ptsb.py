# ptsb_current.py
# PTSB parser:
# - Header: "Date | Details | Withdrawn | Paid In | Balance"
# - Amounts are RIGHT-JUSTIFIED under Withdrawn / Paid In / Balance
# - Opening balance: first "Balance from last stmt"
# - Closing balance: last  "Closing Balance"
# - Ignores right-of-table noise by requiring proximity to column right-edges
# - Transactions shaped like Revolut/BOI JSON

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

from utils import parse_currency  # your existing helper

# ----------------------------
# Regexes / Date parsing
# ----------------------------

# PTSB row dates like "24FEB25" (DDMMMYY or DDMMMYYYY)
RE_DATE_PTSB = re.compile(r"\b(\d{1,2})([A-Za-z]{3})(\d{2,4})\b")

_MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

def _find_date_in_text_ptsb(s: str) -> Optional[str]:
    if not s:
        return None
    m = RE_DATE_PTSB.search(s)
    if not m:
        return None
    d, mon, y = m.groups()
    mon = (mon or "").upper()
    mm = _MONTHS.get(mon)
    if not mm:
        return None
    yy = y
    if len(yy) == 2:
        # assume 20xx
        yy = "20" + yy
    try:
        return f"{int(yy):04d}-{int(mm):02d}-{int(d):02d}"
    except Exception:
        return None

# Row labels we treat specially
RE_BAL_FROM_LAST = re.compile(r"\bBalance\s+from\s+last\s+stmt\b", re.IGNORECASE)
RE_CLOSING_BAL   = re.compile(r"\bClosing\s+Balance\b", re.IGNORECASE)

# ----------------------------
# Header / column detection (PTSB)
# ----------------------------

def _right_edge(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))

def detect_column_positions_ptsb(lines: list[dict]) -> tuple[dict, int, int] | None:
    """
    Locate PTSB headers "Date | Details | Withdrawn | Paid In | Balance".
    Returns (header_positions, header_start_idx, len(lines)) or None.

    We store RIGHT edges for withdrawn/paidin/balance (amounts are right-justified).
    """
    def right_edge(w: dict) -> int:
        return int(w.get("left", 0)) + int(w.get("width", 0))

    for idx, line in enumerate(lines or []):
        words = line.get("words", []) or []
        if not words:
            continue

        pos = {
            "date_left": None,
            "details_left": None,
            "withdrawn_right": None,
            "paidin_right": None,
            "balance_right": None,
        }

        n = len(words)
        for i, w in enumerate(words):
            t = (w.get("text") or "").strip().lower()
            if not t:
                continue

            if t == "date" and pos["date_left"] is None:
                pos["date_left"] = int(w.get("left", 0))

            elif t == "details" and pos["details_left"] is None:
                pos["details_left"] = int(w.get("left", 0))

            elif t in ("withdrawn",) and pos["withdrawn_right"] is None:
                pos["withdrawn_right"] = right_edge(w)

            elif t in ("paid", "paidin", "paid-in", "paidin.", "paid-in.") and pos["paidin_right"] is None:
                pos["paidin_right"] = right_edge(w)
            elif t == "in" and pos["paidin_right"] is None:
                # handle split "Paid In"
                prev = (words[i-1].get("text") or "").strip().lower() if i - 1 >= 0 else ""
                if prev == "paid":
                    pos["paidin_right"] = right_edge(w)

            elif t == "balance" and pos["balance_right"] is None:
                pos["balance_right"] = right_edge(w)

        if all(pos[k] is not None for k in ("date_left", "details_left", "withdrawn_right", "paidin_right", "balance_right")):
            return pos, idx, len(lines)

    return None

def categorise_amount_by_right_edge_ptsb(x_right: int, header_positions: dict, margin: int = 120) -> str:
    """
    Classify token into 'debit' | 'credit' | 'balance' by proximity to the RIGHT edges.
    """
    x_right = int(x_right)
    best = ("unknown", margin + 1)
    for k, label in (("withdrawn_right", "debit"), ("paidin_right", "credit"), ("balance_right", "balance")):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs(x_right - int(pos))
        if d < best[1]:
            best = (label, d)
    return best[0] if best[1] <= margin else "unknown"

# ----------------------------
# Numeric extraction helpers (PTSB)
# ----------------------------

def _rightmost_numeric_within_column(words: list[dict], col_right: int, window: int = 80) -> tuple[Optional[int], Optional[float]]:
    """
    Return (x_right, value) for the numeric whose RIGHT edge is closest to the given column right edge,
    limited to a small window so we don't pick numbers from the info panel to the right of the table.
    """
    best = None  # (abs_delta, xr, value)
    tokens = words or []
    n = len(tokens)

    for i, w in enumerate(tokens):
        t = (w.get("text") or "").strip()
        if not t:
            continue

        v = parse_currency(t, strip_currency=False)
        xr = _right_edge(w)

        # Join with a trailing symbol (rare on PTSB, but harmless)
        if v is None and i + 1 < n:
            nxt = (tokens[i + 1].get("text") or "").strip()
            if nxt == "€":
                v = parse_currency(t + nxt, strip_currency=False)
                xr = _right_edge(tokens[i + 1])

        if v is None:
            continue

        delta = abs(int(xr) - int(col_right))
        if delta <= window:
            if best is None or delta < best[0] or (delta == best[0] and xr > best[1]):
                best = (delta, xr, float(v))

    return (None, None) if best is None else (best[1], best[2])

# ----------------------------
# Opening & Closing balances (PTSB)
# ----------------------------

def extract_opening_closing_ptsb(pages: list[dict]) -> tuple[float | None, float | None]:
    """
    Opening  = first  row whose description matches 'Balance from last stmt' (take its Balance column value)
    Closing  = last   row whose description matches 'Closing Balance'       (take its Balance column value)
    Uses header positions per page and ignores right-of-table noise.
    """
    opening_value: Optional[float] = None
    closing_value: Optional[float] = None

    last_header_positions: Optional[dict] = None

    for page in pages or []:
        lines = page.get("lines", []) or []
        hdr = detect_column_positions_ptsb(lines)
        if hdr:
            last_header_positions = hdr[0]

        pos = last_header_positions
        if not pos:
            continue

        bal_r = int(pos["balance_right"])

        for line in lines:
            ltxt = (line.get("line_text") or "")
            words = line.get("words", []) or []

            if RE_BAL_FROM_LAST.search(ltxt) and opening_value is None:
                # take balance column value near bal_r
                xr, val = _rightmost_numeric_within_column(words, bal_r, window=80)
                if val is not None:
                    opening_value = float(val)

            if RE_CLOSING_BAL.search(ltxt):
                xr, val = _rightmost_numeric_within_column(words, bal_r, window=80)
                if val is not None:
                    closing_value = float(val)  # last one wins across pages

    return opening_value, closing_value

# ----------------------------
# Transactions (PTSB)
# ----------------------------

def parse_transactions_ptsb(
    pages: list[dict],
    iban: str | None = None
) -> list[dict]:
    """
    Parse all PTSB transaction rows across pages. Returns a flat list of transactions.
    - Every row has a date token like 24FEB25 → we use it directly (no carry-forward).
    - Skip "Balance from last stmt" and "Closing Balance" rows.
    - Classify numbers by right-edge proximity to Withdrawn/Paid In/Balance columns.
    - Ignore right-of-table noise by only accepting numbers within a window of column edges.
    """
    all_transactions: List[dict] = []
    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0

    last_header_positions: Optional[dict] = None

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_positions_ptsb(lines)

        if header:
            header_positions, start_idx, end_idx = header
            last_header_positions = header_positions
            line_start = start_idx + 1
            line_end = end_idx
        else:
            if last_header_positions is None:
                continue
            header_positions = last_header_positions
            line_start = 0
            line_end = len(lines)

        bal_r = int(header_positions["balance_right"])
        win = 80  # small window to avoid info table

        for line in lines[line_start: line_end]:
            words = line.get("words", []) or []
            if not words:
                continue

            ltxt = (line.get("line_text") or "")

            # Skip synthetic/opening/closing rows
            if RE_BAL_FROM_LAST.search(ltxt) or RE_CLOSING_BAL.search(ltxt):
                continue

            # Require a date on each row (PTSB format)
            transaction_date = _find_date_in_text_ptsb(ltxt)
            if transaction_date is None:
                # not a transaction row
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue

            df["right"] = df["left"] + df["width"]

            # Only consider numeric-looking tokens
            def _val(x):
                return parse_currency(x, strip_currency=False)

            df["amount_val"] = df["text"].apply(_val)
            df["category"] = df.apply(
                lambda r: categorise_amount_by_right_edge_ptsb(int(r["right"]), header_positions), axis=1
            )

            # For balance, use the restricted window near balance_right to avoid info table
            balance_vals = df.loc[(df["amount_val"].notna()) & (df["category"] == "balance"), ["right", "amount_val"]]
            if not balance_vals.empty:
                # Filter by window
                balance_vals = balance_vals.assign(delta=(balance_vals["right"] - bal_r).abs())
                balance_vals = balance_vals.loc[balance_vals["delta"] <= win]

            withdrawn_vals = df.loc[(df["amount_val"].notna()) & (df["category"] == "debit"), ["right", "amount_val"]]
            paidin_vals    = df.loc[(df["amount_val"].notna()) & (df["category"] == "credit"), ["right", "amount_val"]]

            debit   = float(withdrawn_vals.sort_values("right")["amount_val"].iloc[-1]) if not withdrawn_vals.empty else 0.0
            credit  = float(paidin_vals.sort_values("right")["amount_val"].iloc[-1])    if not paidin_vals.empty    else 0.0
            stmt_balance = float(balance_vals.sort_values("right")["amount_val"].iloc[-1]) if not balance_vals.empty else None

            # If nothing numeric in the amount columns, skip (likely header/separator/noise)
            if debit == 0.0 and credit == 0.0 and stmt_balance is None:
                continue

            # Clean description: remove the date token to keep merchant text clean
            clean_desc = ltxt
            dm = RE_DATE_PTSB.search(clean_desc)
            if dm:
                clean_desc = clean_desc.replace(dm.group(0), "").strip(" |-")

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
                "balance_after_calculated": None,
            })
            seq += 1

    return all_transactions

# ----------------------------
# Build multi-currency sections (from rows)
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
    Opening from rows (fallback): reverse the first transaction against its statement balance (if present).
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
# IBAN (unchanged approach)
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = line.get("line_text", "") or ""
            if "IBAN" in txt.upper():
                after = txt.upper().split("IBAN", 1)[-1]
                cleaned = re.sub(r"\s+", "", after)
                m = iban_pattern.search(cleaned)
                if m:
                    return m.group(1)
    return None

# ----------------------------
# Public entrypoint (PTSB)
# ----------------------------

def parse_statement(raw_ocr, client="Unknown", account_type="Unknown", debug: bool = True):
    def dbg(*a):
        if debug:
            print(*a)

    pages = raw_ocr.get("pages", []) or []

    # IBAN & currency guess
    iban = extract_iban(pages)
    currency_hint = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Transactions (flat)
    transactions = parse_transactions_ptsb(pages, iban=iban)
    dbg(f"🧾 Parsed {len(transactions)} transactions")

    # Group by currency
    buckets = _group_by_currency(transactions)
    dbg("🔑 Buckets from rows:", list(buckets.keys()))

    # Build per-currency sections from rows
    currencies = _build_currency_sections_from_rows(buckets)
    for k, sec0 in currencies.items():
        dbg(f"   - {repr(k)} BEFORE overlay: opening_from_rows={sec0.get('opening_balance')} "
            f"in={sec0.get('money_in_total')} out={sec0.get('money_out_total')}")

    # Explicit opening/closing from labelled rows
    opening_val, closing_val = extract_opening_closing_ptsb(pages)
    dbg(f"📘 Explicit opening_val={opening_val}")
    dbg(f"📘 Explicit closing_val={closing_val}")

    # Choose target currency robustly:
    target_cur = "EUR" if "EUR" in currencies else next(iter(currencies.keys()), currency_hint)
    dbg("🎯 target_cur for overlay:", repr(target_cur))

    if target_cur not in currencies:
        currencies[target_cur] = {
            "opening_balance": None,
            "money_in_total": {"value": 0.0, "currency": target_cur},
            "money_out_total": {"value": 0.0, "currency": target_cur},
            "closing_balance_statement": None,
            "closing_balance_calculated": None,
            "transactions": [],
        }
        dbg(f"ℹ️ Created empty bucket for {repr(target_cur)}")

    sec = currencies[target_cur]

    if opening_val is not None:
        sec["opening_balance"] = {"value": float(round(opening_val, 2)), "currency": target_cur}
        dbg(f"✅ Overlaid OPENING on {repr(target_cur)}:", sec["opening_balance"])
    else:
        dbg("⚠️ No explicit opening to overlay")

    if closing_val is not None:
        sec["closing_balance_statement"] = {"value": float(round(closing_val, 2)), "currency": target_cur}
        dbg(f"✅ Overlaid CLOSING on {repr(target_cur)}:", sec["closing_balance_statement"])
    else:
        dbg("⚠️ No explicit closing to overlay")

    # Recompute a calculated closing if we now have opening + totals
    if sec.get("opening_balance") and sec.get("money_in_total") and sec.get("money_out_total"):
        o = float(sec["opening_balance"]["value"])
        inm = float(sec["money_in_total"]["value"])
        outm = float(sec["money_out_total"]["value"])
        sec["closing_balance_calculated"] = {"value": round(o + inm - outm, 2), "currency": target_cur}
        dbg("🧮 closing_balance_calculated:", sec["closing_balance_calculated"])

    # Statement dates from transactions
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = None
        end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "PTSB",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
